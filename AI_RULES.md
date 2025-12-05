# AI Rules

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: Shadcn/UI (based on Radix UI)
- **Icons**: Lucide React
- **AI Orchestration**: LangGraph & LangChain
- **AI SDK**: Vercel AI SDK (`ai`, `@ai-sdk/openai`)
- **Search/Scraping**: Firecrawl (`@mendable/firecrawl-js`)
- **Toast Notifications**: Sonner

## Development Rules

### Components & Styling
- Always use **Tailwind CSS** for styling. Avoid custom CSS files unless for global styles or complex animations.
- Use **Shadcn/UI** components located in `components/ui/`.
- If you need a new UI primitive, check if it's available in Shadcn/UI before creating a custom one.
- Use **Lucide React** for all icons.

### Project Structure
- `app/`: Next.js App Router pages and layouts.
- `components/`: React components. Put reusable UI components in `components/ui/`.
- `lib/`: Utility functions, AI logic, and configuration.
- `hooks/`: Custom React hooks.

### State & Data Fetching
- Use **React Server Actions** for server-side logic and data mutations.
- Use standard React hooks (`useState`, `useEffect`, `useContext`) for client-side state.
- Keep the `search` function in `app/search.tsx` as a Server Action (`'use server'`).

### AI & Agents
- The core search logic resides in `lib/langgraph-search-engine.ts` using **LangGraph**.
- Web scraping and searching are handled by **Firecrawl** in `lib/firecrawl.ts`.
- Use **Vercel AI SDK** (`ai/rsc`) for streaming responses to the client (e.g., `createStreamableValue`, `readStreamableValue`).

### Best Practices
- Ensure all new code is written in **TypeScript**.
- Prefer **Server Components** by default; add `'use client'` only when necessary (state, effects, event listeners).
- Use `sonner` for toast notifications (`toast.success`, `toast.error`).