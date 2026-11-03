import * as React from "react"
import { cva } from "class-variance-authority"
import { cn } from "../../lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold tracking-wide transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900 shadow-sm",
        secondary:
          "border-transparent bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100",
        destructive:
          "border-transparent bg-rose-500/15 text-rose-700 dark:text-rose-400 border border-rose-500/20",
        outline: "border-slate-200 dark:border-white/10 text-slate-800 dark:text-slate-200",
        blue:
          "border-blue-500/25 bg-blue-500/10 text-blue-700 dark:text-blue-400 font-mono",
        cyan:
          "border-cyan-500/25 bg-cyan-500/10 text-cyan-700 dark:text-cyan-400 font-mono",
        sky:
          "border-sky-500/25 bg-sky-500/10 text-sky-700 dark:text-sky-400 font-mono",
        purple:
          "border-purple-500/25 bg-purple-500/10 text-purple-700 dark:text-purple-400 font-mono",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({ className, variant, ...props }) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
