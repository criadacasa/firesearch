import { StateGraph, END, START, Annotation, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { FirecrawlClient } from './firecrawl';
import { ContextProcessor } from './context-processor';
import { SEARCH_CONFIG, MODEL_CONFIG } from './config';

// Event types remain the same for frontend compatibility
export type SearchPhase = 
  | 'understanding'
  | 'planning' 
  | 'searching'
  | 'analyzing'
  | 'synthesizing'
  | 'complete'
  | 'error';

export type SearchEvent = 
  | { type: 'phase-update'; phase: SearchPhase; message: string }
  | { type: 'thinking'; message: string }
  | { type: 'searching'; query: string; index: number; total: number }
  | { type: 'found'; sources: Source[]; query: string }
  | { type: 'scraping'; url: string; index: number; total: number; query: string }
  | { type: 'content-chunk'; chunk: string }
  | { type: 'final-result'; content: string; sources: Source[]; followUpQuestions?: string[] }
  | { type: 'error'; error: string; errorType?: ErrorType }
  | { type: 'source-processing'; url: string; title: string; stage: 'browsing' | 'extracting' | 'analyzing' }
  | { type: 'source-complete'; url: string; summary: string };

export type ErrorType = 'search' | 'scrape' | 'llm' | 'unknown';

export interface Source {
  url: string;
  title: string;
  content?: string;
  quality?: number;
  summary?: string;
}

export interface SearchResult {
  url: string;
  title: string;
  content?: string;
  markdown?: string;
}

export interface SearchStep {
  id: SearchPhase | string;
  label: string;
  status: 'pending' | 'active' | 'completed';
  startTime?: number;
}

// Proper LangGraph state using Annotation with reducers
const SearchStateAnnotation = Annotation.Root({
  // Input fields
  query: Annotation<string>({
    reducer: (_, y) => y ?? "",
    default: () => ""
  }),
  context: Annotation<{ query: string; response: string }[] | undefined>({
    reducer: (_, y) => y,
    default: () => undefined
  }),
  
  // Process fields
  understanding: Annotation<string | undefined>({
    reducer: (x, y) => y ?? x,
    default: () => undefined
  }),
  searchQueries: Annotation<string[] | undefined>({
    reducer: (x, y) => y ?? x,
    default: () => undefined
  }),
  currentSearchIndex: Annotation<number>({
    reducer: (x, y) => y ?? x,
    default: () => 0
  }),
  
  // Results fields - with proper array reducers
  sources: Annotation<Source[]>({
    reducer: (existing: Source[], update: Source[] | undefined) => {
      if (!update) return existing;
      // Deduplicate sources by URL
      const sourceMap = new Map<string, Source>();
      [...existing, ...update].forEach(source => {
        sourceMap.set(source.url, source);
      });
      return Array.from(sourceMap.values());
    },
    default: () => []
  }),
  scrapedSources: Annotation<Source[]>({
    reducer: (existing: Source[], update: Source[] | undefined) => {
      if (!update) return existing;
      return [...existing, ...update];
    },
    default: () => []
  }),
  processedSources: Annotation<Source[] | undefined>({
    reducer: (x, y) => y ?? x,
    default: () => undefined
  }),
  finalAnswer: Annotation<string | undefined>({
    reducer: (x, y) => y ?? x,
    default: () => undefined
  }),
  followUpQuestions: Annotation<string[] | undefined>({
    reducer: (x, y) => y ?? x,
    default: () => undefined
  }),
  
  // Answer tracking
  subQueries: Annotation<Array<{
    question: string;
    searchQuery: string;
    answered: boolean;
    answer?: string;
    confidence: number;
    sources: string[];
  }> | undefined>({
    reducer: (x, y) => y ?? x,
    default: () => undefined
  }),
  searchAttempt: Annotation<number>({
    reducer: (x, y) => y ?? x,
    default: () => 0
  }),
  
  // Control fields
  phase: Annotation<SearchPhase>({
    reducer: (x, y) => y ?? x,
    default: () => 'understanding' as SearchPhase
  }),
  error: Annotation<string | undefined>({
    reducer: (x, y) => y ?? x,
    default: () => undefined
  }),
  errorType: Annotation<ErrorType | undefined>({
    reducer: (x, y) => y ?? x,
    default: () => undefined
  }),
  maxRetries: Annotation<number>({
    reducer: (x, y) => y ?? x,
    default: () => SEARCH_CONFIG.MAX_RETRIES
  }),
  retryCount: Annotation<number>({
    reducer: (x, y) => y ?? x,
    default: () => 0
  })
});

type SearchState = typeof SearchStateAnnotation.State;

// Define config type for proper event handling
interface GraphConfig {
  configurable?: {
    eventCallback?: (event: SearchEvent) => void;
    checkpointId?: string;
  };
}

export class LangGraphSearchEngine {
  private firecrawl: FirecrawlClient;
  private contextProcessor: ContextProcessor;
  private graph: ReturnType<typeof this.buildGraph>;
  private llm: ChatOpenAI;
  private streamingLlm: ChatOpenAI;
  private checkpointer?: MemorySaver;

  constructor(firecrawl: FirecrawlClient, options?: { enableCheckpointing?: boolean }) {
    this.firecrawl = firecrawl;
    this.contextProcessor = new ContextProcessor();
    
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('OPENAI_API_KEY environment variable is not set');
    }
    
    // Initialize LangChain models
    this.llm = new ChatOpenAI({
      modelName: MODEL_CONFIG.FAST_MODEL,
      temperature: MODEL_CONFIG.TEMPERATURE,
      openAIApiKey: apiKey,
    });
    
    this.streamingLlm = new ChatOpenAI({
      modelName: MODEL_CONFIG.QUALITY_MODEL,
      temperature: MODEL_CONFIG.TEMPERATURE,
      streaming: true,
      openAIApiKey: apiKey,
    });

    // Enable checkpointing if requested
    if (options?.enableCheckpointing) {
      this.checkpointer = new MemorySaver();
    }
    
    this.graph = this.buildGraph();
  }

  getInitialSteps(): SearchStep[] {
    return [
      { id: 'understanding', label: 'Entendendo a solicitação', status: 'pending' },
      { id: 'planning', label: 'Planejando a pesquisa', status: 'pending' },
      { id: 'searching', label: 'Pesquisando fontes', status: 'pending' },
      { id: 'analyzing', label: 'Analisando conteúdo', status: 'pending' },
      { id: 'synthesizing', label: 'Sintetizando resposta', status: 'pending' },
      { id: 'complete', label: 'Concluído', status: 'pending' }
    ];
  }

  private buildGraph() {
    // Create closures for helper methods
    const analyzeQuery = this.analyzeQuery.bind(this);
    const scoreContent = this.scoreContent.bind(this);
    const summarizeContent = this.summarizeContent.bind(this);
    const generateStreamingAnswer = this.generateStreamingAnswer.bind(this);
    const generateFollowUpQuestions = this.generateFollowUpQuestions.bind(this);
    const firecrawl = this.firecrawl;
    const contextProcessor = this.contextProcessor;
    
    const workflow = new StateGraph(SearchStateAnnotation)
      // Understanding node
      .addNode("understand", async (state: SearchState, config?: GraphConfig): Promise<Partial<SearchState>> => {
        const eventCallback = config?.configurable?.eventCallback;
        
        if (eventCallback) {
          eventCallback({
            type: 'phase-update',
            phase: 'understanding',
            message: 'Analisando sua solicitação...'
          });
        }
        
        try {
          const understanding = await analyzeQuery(state.query, state.context);
          
          if (eventCallback) {
            eventCallback({
              type: 'thinking',
              message: understanding
            });
          }
          
          return {
            understanding,
            phase: 'planning' as SearchPhase
          };
        } catch (error) {
          return {
            error: error instanceof Error ? error.message : 'Falha ao entender a solicitação',
            errorType: 'llm' as ErrorType,
            phase: 'error' as SearchPhase
          };
        }
      })
      
      // Planning node
      .addNode("plan", async (state: SearchState, config?: GraphConfig): Promise<Partial<SearchState>> => {
        const eventCallback = config?.configurable?.eventCallback;
        
        if (eventCallback) {
          eventCallback({
            type: 'phase-update',
            phase: 'planning',
            message: 'Planejando estratégia de pesquisa...'
          });
        }
        
        try {
          // Extract sub-queries if not already done
          let subQueries = state.subQueries;
          if (!subQueries) {
            const extractSubQueries = this.extractSubQueries.bind(this);
            const extracted = await extractSubQueries(state.query);
            subQueries = extracted.map(sq => ({
              question: sq.question,
              searchQuery: sq.searchQuery,
              answered: false,
              confidence: 0,
              sources: []
            }));
          }
          
          // Generate search queries for unanswered questions
          const unansweredQueries = subQueries.filter(sq => !sq.answered || sq.confidence < SEARCH_CONFIG.MIN_ANSWER_CONFIDENCE);
          
          if (unansweredQueries.length === 0) {
            // All questions answered, skip to analysis
            return {
              subQueries,
              phase: 'analyzing' as SearchPhase
            };
          }
          
          // Use alternative search queries if this is a retry
          let searchQueries: string[];
          if (state.searchAttempt > 0) {
            const generateAlternativeSearchQueries = this.generateAlternativeSearchQueries.bind(this);
            searchQueries = await generateAlternativeSearchQueries(subQueries, state.searchAttempt);
            
            // Update sub-queries with new search queries
            let alternativeIndex = 0;
            subQueries.forEach(sq => {
              if (!sq.answered || sq.confidence < SEARCH_CONFIG.MIN_ANSWER_CONFIDENCE) {
                if (alternativeIndex < searchQueries.length) {
                  sq.searchQuery = searchQueries[alternativeIndex];
                  alternativeIndex++;
                }
              }
            });
          } else {
            // First attempt - use the search queries from sub-queries
            searchQueries = unansweredQueries.map(sq => sq.searchQuery);
          }
          
          if (eventCallback) {
            if (state.searchAttempt === 0) {
              eventCallback({
                type: 'thinking',
                message: searchQueries.length > 3 
                  ? `Detectei ${subQueries.length} perguntas diferentes. Vou pesquisar cada uma separadamente.`
                  : `Vou pesquisar informações para responder sua pergunta.`
              });
            } else {
              eventCallback({
                type: 'thinking',
                message: `Tentando estratégias alternativas de pesquisa para: ${unansweredQueries.map(sq => sq.question).join(', ')}`
              });
            }
          }
          
          return {
            searchQueries,
            subQueries,
            currentSearchIndex: 0,
            phase: 'searching' as SearchPhase
          };
        } catch (error) {
          return {
            error: error instanceof Error ? error.message : 'Falha ao planejar pesquisa',
            errorType: 'llm' as ErrorType,
            phase: 'error' as SearchPhase
          };
        }
      })
      
      // Search node (handles one search at a time)
      .addNode("search", async (state: SearchState, config?: GraphConfig): Promise<Partial<SearchState>> => {
        const eventCallback = config?.configurable?.eventCallback;
        const searchQueries = state.searchQueries || [];
        const currentIndex = state.currentSearchIndex || 0;
        
        if (currentIndex === 0 && eventCallback) {
          eventCallback({
            type: 'phase-update',
            phase: 'searching',
            message: 'Pesquisando na web...'
          });
        }
        
        if (currentIndex >= searchQueries.length) {
          return {
            phase: 'scrape' as SearchPhase
          };
        }
        
        const searchQuery = searchQueries[currentIndex];
        
        if (eventCallback) {
          eventCallback({
            type: 'searching',
            query: searchQuery,
            index: currentIndex + 1,
            total: searchQueries.length
          });
        }
        
        try {
          const results = await firecrawl.search(searchQuery, {
            limit: SEARCH_CONFIG.MAX_SOURCES_PER_SEARCH,
            scrapeOptions: {
              formats: ['markdown']
            }
          });
          
          const newSources: Source[] = results.data.map((r: SearchResult) => ({
            url: r.url,
            title: r.title,
            content: r.markdown || r.content || '',
            quality: 0
          }));
          
          if (eventCallback) {
            eventCallback({
              type: 'found',
              sources: newSources,
              query: searchQuery
            });
          }
          
          // Process sources in parallel for better performance
          if (SEARCH_CONFIG.PARALLEL_SUMMARY_GENERATION) {
            await Promise.all(newSources.map(async (source) => {
              if (eventCallback) {
                eventCallback({
                  type: 'source-processing',
                  url: source.url,
                  title: source.title,
                  stage: 'browsing'
                });
              }
              
              // Score the content
              source.quality = scoreContent(source.content || '', state.query);
              
              // Generate summary if content is available
              if (source.content && source.content.length > SEARCH_CONFIG.MIN_CONTENT_LENGTH) {
                const summary = await summarizeContent(source.content, searchQuery);
                
                // Store the summary in the source object
                if (summary && !summary.toLowerCase().includes('no specific')) {
                  source.summary = summary;
                  
                  if (eventCallback) {
                    eventCallback({
                      type: 'source-complete',
                      url: source.url,
                      summary: summary
                    });
                  }
                }
              }
            }));
          } else {
            // Original sequential processing
            for (const source of newSources) {
              if (eventCallback) {
                eventCallback({
                  type: 'source-processing',
                  url: source.url,
                  title: source.title,
                  stage: 'browsing'
                });
              }
              
              // Small delay for animation
              await new Promise(resolve => setTimeout(resolve, SEARCH_CONFIG.SOURCE_ANIMATION_DELAY));
              
              // Score the content
              source.quality = scoreContent(source.content || '', state.query);
              
              // Generate summary if content is available
              if (source.content && source.content.length > SEARCH_CONFIG.MIN_CONTENT_LENGTH) {
                const summary = await summarizeContent(source.content, searchQuery);
                
                // Store the summary in the source object
                if (summary && !summary.toLowerCase().includes('no specific')) {
                  source.summary = summary;
                  
                  if (eventCallback) {
                    eventCallback({
                      type: 'source-complete',
                      url: source.url,
                      summary: summary
                    });
                  }
                }
              }
            }
          }
          
          return {
            sources: newSources,
            currentSearchIndex: currentIndex + 1
          };
        } catch {
          return {
            currentSearchIndex: currentIndex + 1,
            errorType: 'search' as ErrorType
          };
        }
      })
      
      // Scraping node
      .addNode("scrape", async (state: SearchState, config?: GraphConfig): Promise<Partial<SearchState>> => {
        const eventCallback = config?.configurable?.eventCallback;
        const sourcesToScrape = state.sources?.filter(s => 
          !s.content || s.content.length < SEARCH_CONFIG.MIN_CONTENT_LENGTH
        ) || [];
        const newScrapedSources: Source[] = [];
        
        // Sources with content were already processed in search node, just pass them through
        const sourcesWithContent = state.sources?.filter(s => 
          s.content && s.content.length >= SEARCH_CONFIG.MIN_CONTENT_LENGTH
        ) || [];
        newScrapedSources.push(...sourcesWithContent);
        
        // Then scrape sources without content
        for (let i = 0; i < Math.min(sourcesToScrape.length, SEARCH_CONFIG.MAX_SOURCES_TO_SCRAPE); i++) {
          const source = sourcesToScrape[i];
          
          if (eventCallback) {
            eventCallback({
              type: 'scraping',
              url: source.url,
              index: newScrapedSources.length + 1,
              total: sourcesWithContent.length + Math.min(sourcesToScrape.length, SEARCH_CONFIG.MAX_SOURCES_TO_SCRAPE),
              query: state.query
            });
          }
          
          try {
            const scraped = await firecrawl.scrapeUrl(source.url, SEARCH_CONFIG.SCRAPE_TIMEOUT);
            if (scraped.success && scraped.markdown) {
              const enrichedSource = {
                ...source,
                content: scraped.markdown,
                quality: scoreContent(scraped.markdown, state.query)
              };
              newScrapedSources.push(enrichedSource);
              
              // Show processing animation
              if (eventCallback) {
                eventCallback({
                  type: 'source-processing',
                  url: source.url,
                  title: source.title,
                  stage: 'browsing'
                });
              }
              
              await new Promise(resolve => setTimeout(resolve, 150));
              
              const summary = await summarizeContent(scraped.markdown, state.query);
              if (summary) {
                enrichedSource.summary = summary;
                
                if (eventCallback) {
                  eventCallback({
                    type: 'source-complete',
                    url: source.url,
                    summary: summary
                  });
                }
              }
            } else if (scraped.error === 'timeout') {
              if (eventCallback) {
                eventCallback({
                  type: 'thinking',
                  message: `${new URL(source.url).hostname} está demorando muito, pulando...`
                });
              }
            }
          } catch {
            if (eventCallback) {
              eventCallback({
                type: 'thinking',
                message: `Não foi possível acessar ${new URL(source.url).hostname}, tentando outras fontes...`
              });
            }
          }
        }
        
        return {
          scrapedSources: newScrapedSources,
          phase: 'analyzing' as SearchPhase
        };
      })
      
      // Analyzing node
      .addNode("analyze", async (state: SearchState, config?: GraphConfig): Promise<Partial<SearchState>> => {
        const eventCallback = config?.configurable?.eventCallback;
        
        if (eventCallback) {
          eventCallback({
            type: 'phase-update',
            phase: 'analyzing',
            message: 'Analisando informações coletadas...'
          });
        }
        
        // Combine sources and remove duplicates by URL
        const sourceMap = new Map<string, Source>();
        
        // Add all sources (not just those with long content, since summaries contain key info)
        (state.sources || []).forEach(s => sourceMap.set(s.url, s));
        
        // Add scraped sources (may override with better content)
        (state.scrapedSources || []).forEach(s => sourceMap.set(s.url, s));
        
        const allSources = Array.from(sourceMap.values());
        
        // Check which questions have been answered
        if (state.subQueries) {
          const checkAnswersInSources = this.checkAnswersInSources.bind(this);
          const updatedSubQueries = await checkAnswersInSources(state.subQueries, allSources);
          
          const answeredCount = updatedSubQueries.filter(sq => sq.answered).length;
          const totalQuestions = updatedSubQueries.length;
          const searchAttempt = (state.searchAttempt || 0) + 1;
          
          // Check if we have partial answers with decent confidence
          const partialAnswers = updatedSubQueries.filter(sq => sq.confidence >= 0.3);
          const hasPartialInfo = partialAnswers.length > answeredCount;
          
          if (eventCallback) {
            if (answeredCount === totalQuestions) {
              eventCallback({
                type: 'thinking',
                message: `Encontradas respostas para todas as ${totalQuestions} perguntas em ${allSources.length} fontes`
              });
            } else if (answeredCount > 0) {
              eventCallback({
                type: 'thinking',
                message: `Encontradas respostas para ${answeredCount} de ${totalQuestions} perguntas. Ainda faltando: ${updatedSubQueries.filter(sq => !sq.answered).map(sq => sq.question).join(', ')}`
              });
            } else if (searchAttempt >= SEARCH_CONFIG.MAX_SEARCH_ATTEMPTS) {
              // Only show "could not find" message when we've exhausted all attempts
              eventCallback({
                type: 'thinking',
                message: `Não foi possível encontrar respostas específicas em ${allSources.length} fontes. A informação pode não estar disponível publicamente.`
              });
            } else if (hasPartialInfo && searchAttempt >= 3) {
              // If we have partial info and tried 3+ times, stop searching
              eventCallback({
                type: 'thinking',
                message: `Encontradas informações parciais. Prosseguindo com o que está disponível.`
              });
            } else {
              // For intermediate attempts, show a different message
              eventCallback({
                type: 'thinking',
                message: `Pesquisando por informações mais específicas...`
              });
            }
          }
          
          // If we haven't found all answers and haven't exceeded attempts, try again
          // BUT stop if we have partial info and already tried 2+ times
          if (answeredCount < totalQuestions && 
              searchAttempt < SEARCH_CONFIG.MAX_SEARCH_ATTEMPTS &&
              !(hasPartialInfo && searchAttempt >= 2)) {
            return {
              sources: allSources,
              subQueries: updatedSubQueries,
              searchAttempt,
              phase: 'planning' as SearchPhase  // Go back to planning for retry
            };
          }
          
          // Otherwise proceed with what we have
          try {
            const processedSources = await contextProcessor.processSources(
              state.query,
              allSources,
              state.searchQueries || []
            );
            
            return {
              sources: allSources,
              processedSources,
              subQueries: updatedSubQueries,
              searchAttempt,
              phase: 'synthesizing' as SearchPhase
            };
          } catch {
            return {
              sources: allSources,
              processedSources: allSources,
              subQueries: updatedSubQueries,
              searchAttempt,
              phase: 'synthesizing' as SearchPhase
            };
          }
        } else {
          // Fallback for queries without sub-queries
          if (eventCallback && allSources.length > 0) {
            eventCallback({
              type: 'thinking',
              message: `Encontradas ${allSources.length} fontes com informações de qualidade`
            });
          }
          
          try {
            const processedSources = await contextProcessor.processSources(
              state.query,
              allSources,
              state.searchQueries || []
            );
            
            return {
              sources: allSources,
              processedSources,
              phase: 'synthesizing' as SearchPhase
            };
          } catch {
            return {
              sources: allSources,
              processedSources: allSources,
              phase: 'synthesizing' as SearchPhase
            };
          }
        }
      })
      
      // Synthesizing node with streaming
      .addNode("synthesize", async (state: SearchState, config?: GraphConfig): Promise<Partial<SearchState>> => {
        const eventCallback = config?.configurable?.eventCallback;
        
        if (eventCallback) {
          eventCallback({
            type: 'phase-update',
            phase: 'synthesizing',
            message: 'Criando resposta abrangente...'
          });
        }
        
        try {
          const sourcesToUse = state.processedSources || state.sources || [];
          
          const answer = await generateStreamingAnswer(
            state.query,
            sourcesToUse,
            (chunk) => {
              if (eventCallback) {
                eventCallback({ type: 'content-chunk', chunk });
              }
            },
            state.context
          );
          
          // Generate follow-up questions
          const followUpQuestions = await generateFollowUpQuestions(
            state.query,
            answer,
            sourcesToUse,
            state.context
          );
          
          return {
            finalAnswer: answer,
            followUpQuestions,
            phase: 'complete' as SearchPhase
          };
        } catch (error) {
          return {
            error: error instanceof Error ? error.message : 'Falha ao gerar resposta',
            errorType: 'llm' as ErrorType,
            phase: 'error' as SearchPhase
          };
        }
      })
      
      // Error handling node
      .addNode("handleError", async (state: SearchState, config?: GraphConfig): Promise<Partial<SearchState>> => {
        const eventCallback = config?.configurable?.eventCallback;
        
        if (eventCallback) {
          eventCallback({
            type: 'error',
            error: state.error || 'Ocorreu um erro desconhecido',
            errorType: state.errorType
          });
        }
        
        // Retry logic based on error type
        if ((state.retryCount || 0) < (state.maxRetries || SEARCH_CONFIG.MAX_RETRIES)) {
              
          // Different retry strategies based on error type
          const retryPhase = state.errorType === 'search' ? 'searching' : 'understanding';
          
          return {
            retryCount: (state.retryCount || 0) + 1,
            phase: retryPhase as SearchPhase,
            error: undefined,
            errorType: undefined
          };
        }
        
        return {
          phase: 'error' as SearchPhase
        };
      })
      
      // Complete node
      .addNode("complete", async (state: SearchState, config?: GraphConfig): Promise<Partial<SearchState>> => {
        const eventCallback = config?.configurable?.eventCallback;
        
        if (eventCallback) {
          eventCallback({
            type: 'phase-update',
            phase: 'complete',
            message: 'Pesquisa concluída!'
          });
          
          eventCallback({
            type: 'final-result',
            content: state.finalAnswer || '',
            sources: state.sources || [],
            followUpQuestions: state.followUpQuestions
          });
        }
        
        return {
          phase: 'complete' as SearchPhase
        };
      });

    // Add edges with proper conditional routing
    workflow
      .addEdge(START, "understand")
      .addConditionalEdges(
        "understand",
        (state: SearchState) => state.phase === 'error' ? "handleError" : "plan",
        {
          handleError: "handleError",
          plan: "plan"
        }
      )
      .addConditionalEdges(
        "plan",
        (state: SearchState) => state.phase === 'error' ? "handleError" : "search",
        {
          handleError: "handleError",
          search: "search"
        }
      )
      .addConditionalEdges(
        "search",
        (state: SearchState) => {
          if (state.phase === 'error') return "handleError";
          if ((state.currentSearchIndex || 0) < (state.searchQueries?.length || 0)) {
            return "search"; // Continue searching
          }
          return "scrape"; // Move to scraping
        },
        {
          handleError: "handleError",
          search: "search",
          scrape: "scrape"
        }
      )
      .addConditionalEdges(
        "scrape",
        (state: SearchState) => state.phase === 'error' ? "handleError" : "analyze",
        {
          handleError: "handleError",
          analyze: "analyze"
        }
      )
      .addConditionalEdges(
        "analyze",
        (state: SearchState) => {
          if (state.phase === 'error') return "handleError";
          if (state.phase === 'planning') return "plan";  // Retry with new searches
          return "synthesize";
        },
        {
          handleError: "handleError",
          plan: "plan",
          synthesize: "synthesize"
        }
      )
      .addConditionalEdges(
        "synthesize",
        (state: SearchState) => state.phase === 'error' ? "handleError" : "complete",
        {
          handleError: "handleError",
          complete: "complete"
        }
      )
      .addConditionalEdges(
        "handleError",
        (state: SearchState) => state.phase === 'error' ? END : "understand",
        {
          [END]: END,
          understand: "understand"
        }
      )
      .addEdge("complete", END);

    // Compile with optional checkpointing
    return workflow.compile(this.checkpointer ? { checkpointer: this.checkpointer } : undefined);
  }

  async search(
    query: string,
    onEvent: (event: SearchEvent) => void,
    context?: { query: string; response: string }[],
    checkpointId?: string
  ): Promise<void> {
    try {
      const initialState: SearchState = {
        query,
        context,
        sources: [],
        scrapedSources: [],
        processedSources: undefined,
        phase: 'understanding',
        currentSearchIndex: 0,
        maxRetries: SEARCH_CONFIG.MAX_RETRIES,
        retryCount: 0,
        understanding: undefined,
        searchQueries: undefined,
        finalAnswer: undefined,
        followUpQuestions: undefined,
        error: undefined,
        errorType: undefined,
        subQueries: undefined,
        searchAttempt: 0
      };

      // Configure with event callback
      const config: GraphConfig = {
        configurable: {
          eventCallback: onEvent,
          ...(checkpointId && this.checkpointer ? { thread_id: checkpointId } : {})
        }
      };

      // Invoke the graph with increased recursion limit
      await this.graph.invoke(initialState, {
        ...config,
        recursionLimit: 35  // Increased from default 25 to handle MAX_SEARCH_ATTEMPTS=5
      });
    } catch (error) {
      onEvent({
        type: 'error',
        error: error instanceof Error ? error.message : 'A pesquisa falhou',
        errorType: 'unknown'
      });
    }
  }


  // Get current date for context
  private getCurrentDateContext(): string {
    const now = new Date();
    const dateStr = now.toLocaleDateString('pt-BR', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
    
    return `A data de hoje é ${dateStr}.`;
  }

  // Pure helper methods (no side effects)
  private async analyzeQuery(query: string, context?: { query: string; response: string }[]): Promise<string> {
    let contextPrompt = '';
    if (context && context.length > 0) {
      contextPrompt = '\n\nConversa anterior:\n';
      context.forEach(c => {
        contextPrompt += `Usuário: ${c.query}\nAssistente: ${c.response.substring(0, SEARCH_CONFIG.CONTEXT_PREVIEW_LENGTH)}...\n\n`;
      });
    }
    
    const messages = [
      new SystemMessage(`${this.getCurrentDateContext()}

Analise esta consulta de pesquisa e explique em PORTUGUÊS o que você entendeu que o usuário está procurando.

Instruções:
- Comece com um título claro e conciso (ex: "Pesquisando sobre escassez de ovos" ou "Entendendo os impactos da mudança climática")
- Depois explique em 1-2 frases quais aspectos do tópico o usuário quer saber
- Se isso se relacionar com perguntas anteriores, reconheça essa conexão
- Finalmente, mencione que você pesquisará informações para ajudar a responder a pergunta
- Só mencione pesquisar por informações "mais recentes" se a consulta for explicitamente sobre eventos recentes ou tendências atuais

Mantenha natural e conversacional, mostrando que você realmente entendeu o pedido. RESPONDA SEMPRE EM PORTUGUÊS.`),
      new HumanMessage(`Consulta: "${query}"${contextPrompt}`)
    ];
    
    const response = await this.llm.invoke(messages);
    return response.content.toString();
  }

  private async checkAnswersInSources(
    subQueries: Array<{ question: string; searchQuery: string; answered: boolean; answer?: string; confidence: number; sources: string[] }>,
    sources: Source[]
  ): Promise<typeof subQueries> {
    if (sources.length === 0) return subQueries;
    
    const messages = [
      new SystemMessage(`Verifique quais perguntas foram respondidas pelas fontes fornecidas.

Para cada pergunta, determine:
1. Se as fontes contêm uma resposta direta
2. O nível de confiança (0.0-1.0) de que a pergunta foi totalmente respondida
3. Um breve resumo da resposta se encontrada (EM PORTUGUÊS)

Diretrizes:
- Para perguntas "quem" sobre pessoas/fundadores: Marque como respondido (confiança 0.8+) se encontrar nomes de pessoas específicas
- Para perguntas "o que": Marque como respondido (confiança 0.8+) se encontrar as informações específicas solicitadas
- Para perguntas "quando": Marque como respondido (confiança 0.8+) se encontrar datas ou períodos
- Para perguntas "quantos": Exija números específicos (confiança 0.8+)
- Para perguntas de comparação: Exija informações sobre todos os itens sendo comparados
- Se as fontes respondem claramente à pergunta mas faltam alguns detalhes menores, use confiança média (0.6-0.7)
- Se as fontes mencionam o tópico mas não respondem à pergunta específica, use confiança baixa (< 0.3)

Retorne APENAS um array JSON, sem formatação markdown ou blocos de código:
[
  {
    "question": "a pergunta original",
    "answered": true/false,
    "confidence": 0.0-1.0,
    "answer": "breve resposta se encontrada",
    "sources": ["urls que contêm a resposta"]
  }
]`),
      new HumanMessage(`Perguntas para verificar:
${subQueries.map(sq => sq.question).join('\n')}

Fontes:
${sources.slice(0, SEARCH_CONFIG.MAX_SOURCES_TO_CHECK).map(s => {
  let sourceInfo = `URL: ${s.url}\nTítulo: ${s.title}\n`;
  
  // Include summary if available (this is the key insight from the search)
  if (s.summary) {
    sourceInfo += `Resumo: ${s.summary}\n`;
  }
  
  // Include content preview
  if (s.content) {
    sourceInfo += `Conteúdo: ${s.content.slice(0, SEARCH_CONFIG.ANSWER_CHECK_PREVIEW)}\n`;
  }
  
  return sourceInfo;
}).join('\n---\n')}`)
    ];

    try {
      const response = await this.llm.invoke(messages);
      let content = response.content.toString();
      
      // Strip markdown code blocks if present
      content = content.replace(/```json\s*/g, '').replace(/```\s*$/g, '').trim();
      
      const results = JSON.parse(content);
      
      // Update sub-queries with results
      return subQueries.map(sq => {
        const result = results.find((r: { question: string }) => r.question === sq.question);
        if (result && result.confidence > sq.confidence) {
          return {
            ...sq,
            answered: result.confidence >= SEARCH_CONFIG.MIN_ANSWER_CONFIDENCE,
            answer: result.answer,
            confidence: result.confidence,
            sources: [...new Set([...sq.sources, ...(result.sources || [])])]
          };
        }
        return sq;
      });
    } catch (error) {
      console.error('Error checking answers:', error);
      return subQueries;
    }
  }

  private async extractSubQueries(query: string): Promise<Array<{ question: string; searchQuery: string }>> {
    const messages = [
      new SystemMessage(`Extraia as perguntas factuais individuais desta consulta. Cada pergunta deve ser algo que possa ser definitivamente respondido.

IMPORTANTE: 
- Quando o usuário menciona algo com uma versão/número (como "deepseek r1 0528"), inclua a versão COMPLETA na pergunta
- Para a consulta de pesquisa (searchQuery), você pode simplificar um pouco mas mantenha os identificadores principais
- Use PORTUGUÊS para as perguntas extraídas, mas pode usar termos em inglês para searchQuery se for mais provável encontrar resultados (termos técnicos, nomes de produtos)

Exemplos:
"Quem fundou a Anthropic e quando" → 
[
  {"question": "Quem fundou a Anthropic?", "searchQuery": "Anthropic founders"},
  {"question": "Quando a Anthropic foi fundada?", "searchQuery": "Anthropic founded date year"}
]

Importante: 
- Para pedidos de comparação, crie uma única pergunta/pesquisa que cubra ambos os itens
- Mantenha o número de sub-consultas razoável (mire em 3-5 no máximo)

Retorne APENAS um array JSON de objetos {question, searchQuery}.`),
      new HumanMessage(`Consulta: "${query}"`)
    ];

    try {
      const response = await this.llm.invoke(messages);
      return JSON.parse(response.content.toString());
    } catch {
      // Fallback: treat as single query
      return [{ question: query, searchQuery: query }];
    }
  }

  private async generateAlternativeSearchQueries(
    subQueries: Array<{ question: string; searchQuery: string; answered: boolean; answer?: string; confidence: number; sources: string[] }>,
    previousAttempts: number
  ): Promise<string[]> {
    const unansweredQueries = subQueries.filter(sq => !sq.answered || sq.confidence < SEARCH_CONFIG.MIN_ANSWER_CONFIDENCE);
    
    // If we're on attempt 3 and still searching for the same thing, just give up on that specific query
    if (previousAttempts >= 2) {
      const problematicQueries = unansweredQueries.filter(sq => {
        // Check if the question contains a version number or specific identifier that might not exist
        const hasVersionPattern = /\b\d{3,4}\b|\bv\d+\.\d+|\bversion\s+\d+/i.test(sq.question);
        const hasFailedMultipleTimes = previousAttempts >= 2;
        return hasVersionPattern && hasFailedMultipleTimes;
      });
      
      if (problematicQueries.length > 0) {
        // Return generic searches that might find partial info
        return problematicQueries.map(sq => {
          const baseTerm = sq.question.replace(/0528|specific version/gi, '').trim();
          return baseTerm.substring(0, 50); // Keep it short
        });
      }
    }
    
    const messages = [
      new SystemMessage(`${this.getCurrentDateContext()}

Gere consultas de pesquisa ALTERNATIVAS para perguntas que não foram respondidas nas tentativas anteriores.

Tentativas de pesquisa anteriores: ${previousAttempts}
Consultas anteriores que não encontraram respostas:
${unansweredQueries.map(sq => `- Pergunta: "${sq.question}"\n  Pesquisa anterior: "${sq.searchQuery}"`).join('\n')}

IMPORTANTE: Se estiver pesquisando por algo com uma versão/data específica que continua falhando, tente pesquisar apenas pelo produto base sem a versão.

Gere NOVAS consultas de pesquisa usando estas estratégias:
1. Tente termos mais amplos ou gerais
2. Tente frases diferentes ou sinônimos
3. Remova qualificadores específicos (como anos ou versões) se forem muito restritivos
4. Tente pesquisar por conceitos relacionados que possam conter a resposta
5. Para produtos que podem não existir, pesquise pela empresa ou nome do produto base

Retorne uma consulta de pesquisa alternativa por pergunta não respondida, uma por linha.`),
      new HumanMessage(`Gere pesquisas alternativas para estas ${unansweredQueries.length} perguntas não respondidas.`)
    ];

    try {
      const response = await this.llm.invoke(messages);
      const result = response.content.toString();
      
      const queries = result
        .split('\n')
        .map(q => q.trim())
        .map(q => q.replace(/^["']|["']$/g, ''))
        .map(q => q.replace(/^\d+\.\s*/, ''))
        .map(q => q.replace(/^[-*#]\s*/, ''))
        .filter(q => q.length > 0)
        .filter(q => !q.match(/^```/))
        .filter(q => q.length > 3);
      
      return queries.slice(0, SEARCH_CONFIG.MAX_SEARCH_QUERIES);
    } catch {
      // Fallback: return original queries with slight modifications
      return unansweredQueries.map(sq => sq.searchQuery + " news reports").slice(0, SEARCH_CONFIG.MAX_SEARCH_QUERIES);
    }
  }

  private scoreContent(content: string, query: string): number {
    const queryWords = query.toLowerCase().split(' ');
    const contentLower = content.toLowerCase();
    
    let score = 0;
    for (const word of queryWords) {
      if (contentLower.includes(word)) score += 0.2;
    }
    
    return Math.min(score, 1);
  }

  private async summarizeContent(content: string, query: string): Promise<string> {
    try {
      const messages = [
        new SystemMessage(`${this.getCurrentDateContext()}

Extraia UMA descoberta chave deste conteúdo que seja ESPECIFICAMENTE relevante para a consulta de pesquisa. RESPONDA EM PORTUGUÊS.

CRÍTICO: Apenas resuma informações que se relacionam diretamente com a consulta de pesquisa.
- Se nenhuma informação relevante for encontrada, apenas retorne o fato mais relevante da página

Instruções:
- Retorne apenas UMA frase com uma descoberta específica em PORTUGUÊS
- Inclua números, datas ou detalhes específicos quando disponíveis
- Mantenha abaixo de ${SEARCH_CONFIG.SUMMARY_CHAR_LIMIT} caracteres
- Não diga "Nenhuma informação relevante foi encontrada" - encontre algo relevante para a pesquisa atual`),
        new HumanMessage(`Consulta: "${query}"\n\nConteúdo: ${content.slice(0, 2000)}`)
      ];
      
      const response = await this.llm.invoke(messages);
      return response.content.toString().trim();
    } catch {
      return '';
    }
  }

  private async generateStreamingAnswer(
    query: string,
    sources: Source[],
    onChunk: (chunk: string) => void,
    context?: { query: string; response: string }[]
  ): Promise<string> {
    const sourcesText = sources
      .map((s, i) => {
        if (!s.content) return `[${i + 1}] ${s.title}\n[Sem conteúdo disponível]`;
        return `[${i + 1}] ${s.title}\n${s.content}`;
      })
      .join('\n\n');
    
    let contextPrompt = '';
    if (context && context.length > 0) {
      contextPrompt = '\n\nConversa anterior para contexto:\n';
      context.forEach(c => {
        contextPrompt += `Usuário: ${c.query}\nAssistente: ${c.response.substring(0, 300)}...\n\n`;
      });
    }
    
    const messages = [
      new SystemMessage(`${this.getCurrentDateContext()}

Responda à pergunta do usuário EM PORTUGUÊS DO BRASIL com base nas fontes fornecidas. Forneça uma resposta clara e abrangente com citações [1], [2], etc. Use formatação markdown para melhor legibilidade. Se esta pergunta se relaciona a tópicos discutidos anteriormente, faça conexões onde for relevante.

Sempre use o formato de citação [1] no texto quando afirmar fatos das fontes.`),
      new HumanMessage(`Pergunta: "${query}"${contextPrompt}\n\nBaseado nestas fontes:\n${sourcesText}`)
    ];
    
    let fullText = '';
    
    try {
      const stream = await this.streamingLlm.stream(messages);
      
      for await (const chunk of stream) {
        const content = chunk.content;
        if (typeof content === 'string') {
          fullText += content;
          onChunk(content);
        }
      }
    } catch {
      // Fallback to non-streaming if streaming fails
      const response = await this.llm.invoke(messages);
      fullText = response.content.toString();
      onChunk(fullText);
    }
    
    return fullText;
  }

  private async generateFollowUpQuestions(
    originalQuery: string,
    answer: string,
    _sources: Source[],
    context?: { query: string; response: string }[]
  ): Promise<string[]> {
    try {
      let contextPrompt = '';
      if (context && context.length > 0) {
        contextPrompt = '\n\nTópicos da conversa anterior:\n';
        context.forEach(c => {
          contextPrompt += `- ${c.query}\n`;
        });
        contextPrompt += '\nConsidere o fluxo completo da conversa ao gerar o acompanhamento.\n';
      }
      
      const messages = [
        new SystemMessage(`${this.getCurrentDateContext()}

Com base nesta consulta de pesquisa e resposta, gere 3 perguntas de acompanhamento relevantes EM PORTUGUÊS que o usuário pode querer explorar a seguir.

Instruções:
- Gere exatamente 3 perguntas de acompanhamento
- Cada pergunta deve explorar um aspecto diferente ou se aprofundar no tópico
- As perguntas devem ser naturais e conversacionais
- Elas devem se basear nas informações fornecidas na resposta
- Torne-as específicas e acionáveis
- Mantenha cada pergunta com menos de 80 caracteres
- Retorne apenas as perguntas, uma por linha, sem numeração ou marcadores
- Considere todo o contexto da conversa ao gerar perguntas

Exemplos de boas perguntas de acompanhamento:
- "Como isso se compara ao [alternativa]?"
- "Você pode explicar [termo técnico] com mais detalhes?"
- "Quais são as aplicações práticas disso?"
- "Quais são os principais benefícios e desvantagens?"
- "Como isso é tipicamente implementado?"`),
        new HumanMessage(`Consulta original: "${originalQuery}"\n\nResumo da resposta: ${answer.length > 1000 ? answer.slice(0, 1000) + '...' : answer}${contextPrompt}`)
      ];
      
      const response = await this.llm.invoke(messages);
      const questions = response.content.toString()
        .split('\n')
        .map(q => q.trim())
        .filter(q => q.length > 0 && q.length < 80)
        .slice(0, 3);
      
      return questions.length > 0 ? questions : [];
    } catch {
      return [];
    }
  }
}