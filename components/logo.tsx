"use client";

import React from 'react';

export function Logo() {
  return (
    <div className="flex items-center gap-3 select-none">
      <svg 
        viewBox="0 0 100 100" 
        className="w-10 h-10 flex-shrink-0"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-label="DeepTarget Logo"
      >
        {/* Outer Ring - Blue */}
        <circle cx="50" cy="50" r="40" stroke="#1e40af" strokeWidth="8" className="dark:stroke-blue-500" />
        
        {/* Middle Ring - Gray */}
        <circle cx="50" cy="50" r="26" stroke="#9ca3af" strokeWidth="8" className="dark:stroke-gray-500" />
        
        {/* Inner Dot - Blue */}
        <circle cx="50" cy="50" r="10" fill="#1e40af" className="dark:fill-blue-500" />
        
        {/* Arrow Shaft (White outline for separation + Blue shaft) */}
        <path d="M50 50 L82 18" stroke="white" strokeWidth="12" className="dark:stroke-gray-950" strokeLinecap="round" />
        <path d="M50 50 L82 18" stroke="#1e40af" strokeWidth="8" className="dark:stroke-blue-500" strokeLinecap="round" />
        
        {/* Arrow Fletching */}
        <path d="M82 18 L92 18 L82 8 Z" fill="#1e40af" className="dark:fill-blue-500" />
      </svg>
      
      <div className="flex flex-col justify-center -space-y-0.5">
        <span className="font-bold text-2xl leading-none text-[#1e40af] dark:text-blue-500 tracking-tight">
          DeepTarget
        </span>
        <span className="text-[0.6rem] font-bold tracking-[0.15em] text-[#1e40af] dark:text-blue-500 uppercase">
          AI Research App
        </span>
      </div>
    </div>
  );
}