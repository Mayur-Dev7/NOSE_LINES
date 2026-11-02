import React from "react"
import { Sun, Moon } from "lucide-react"
import { useTheme } from "../theme-provider"
import { cn } from "../../lib/utils"

export function ThemeToggle({ className }) {
  const { theme, toggleTheme } = useTheme()
  const isDark = theme === "dark"

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className={cn(
        "relative inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full",
        "bg-slate-200/80 dark:bg-slate-800/80 hover:bg-slate-300/80 dark:hover:bg-slate-700/80",
        "border border-black/5 dark:border-white/10 shadow-sm",
        "text-xs font-medium text-slate-700 dark:text-slate-200",
        "transition-all duration-300 active:scale-95 cursor-pointer backdrop-blur-md select-none",
        className
      )}
      title={`Switch to ${isDark ? "Light" : "Dark"} Mode`}
      aria-label="Toggle theme"
    >
      <div className="relative flex items-center justify-center w-4 h-4">
        {isDark ? (
          <Moon className="w-3.5 h-3.5 text-blue-400 transition-transform duration-300 rotate-0" />
        ) : (
          <Sun className="w-3.5 h-3.5 text-amber-500 transition-transform duration-300 rotate-0" />
        )}
      </div>
      <span className="capitalize tracking-tight text-[11px] font-medium">
        {isDark ? "Dark" : "Light"}
      </span>
    </button>
  )
}
