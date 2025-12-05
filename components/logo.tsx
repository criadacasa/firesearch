"use client";

import React, { useState } from 'react';
import { Target } from 'lucide-react';

export function Logo() {
  const [hasError, setHasError] = useState(false);

  if (hasError) {
    return (
      <div className="flex items-center gap-2 font-bold text-xl text-[#36322F] dark:text-white">
         <Target className="w-6 h-6 text-red-600" />
         <span>DeepTarget</span>
      </div>
    );
  }

  return (
    <img
      src="https://drive.google.com/uc?export=view&id=1F4CX-akjLBblGm4Ad21LAHPrCFnVUhEe"
      alt="DeepTarget Logo"
      className="h-8 w-auto object-contain"
      onError={() => setHasError(true)}
    />
  );
}