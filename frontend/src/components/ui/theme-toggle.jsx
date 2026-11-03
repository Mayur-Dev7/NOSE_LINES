import React from "react"
import { Sun, Moon } from "lucide-react"
import { useTheme } from "../theme-provider"
import { cn } from "../../lib/utils"

export function ThemeToggle({ className }) {
  const { theme, setTheme } = useTheme()
  const isDark = theme === "dark"

  return (
    <div
      className={cn(
        "inline-flex items-center p-0.5 rounded-full",
        "bg-slate-200/90 dark:bg-slate-900/90",
        "border border-black/[0.08] dark:border-white/[0.1] shadow-inner",
        className
      )}
      role="group"
      aria-label="Theme selector"
    >
      <button
        type="button"
        onClick={() => setTheme("light")}
        className={cn(
          "flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-all cursor-pointer select-none",
          !isDark
            ? "bg-white text-slate-900 shadow-sm border border-black/[0.06] font-semibold"
            : "text-slate-500 hover:text-slate-900 dark:hover:text-slate-200"
        )}
      >
        <Sun className={cn("w-3.5 h-3.5", !isDark ? "text-amber-500" : "text-slate-400")} />
        <span>Light</span>
      </button>
      <button
        type="button"
        onClick={() => setTheme("dark")}
        className={cn(
          "flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-all cursor-pointer select-none",
          isDark
            ? "bg-slate-800 text-white shadow-sm border border-white/10 font-semibold"
            : "text-slate-500 hover:text-slate-900 dark:hover:text-slate-200"
        )}
      >
        <Moon className={cn("w-3.5 h-3.5", isDark ? "text-blue-400" : "text-slate-400")} />
        <span>Dark</span>
      </button>
    </div>
  )
}
