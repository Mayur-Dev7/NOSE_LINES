import React, { createContext, useContext, useEffect, useState } from "react"

const ThemeContext = createContext({
  theme: "dark",
  setTheme: () => null,
  toggleTheme: () => null,
})

export function ThemeProvider({
  children,
  defaultTheme = "dark",
  storageKey = "bovine-theme",
}) {
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem(storageKey)
      if (saved === "dark" || saved === "light") return saved
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
        return "light"
      }
    } catch (e) {}
    return defaultTheme
  })

  useEffect(() => {
    const root = window.document.documentElement
    root.classList.remove("light", "dark")
    root.classList.add(theme)
    root.style.colorScheme = theme
    try {
      localStorage.setItem(storageKey, theme)
    } catch (e) {}
  }, [theme, storageKey])

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"))
  }

  return (
    <ThemeContext.Provider value={{ theme, setTheme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export const useTheme = () => {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider")
  }
  return context
}
