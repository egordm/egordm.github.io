// Always use dark mode - no toggle, no system preference
document.documentElement.setAttribute("saved-theme", "dark")
localStorage.setItem("theme", "dark")

const emitThemeChangeEvent = (theme: "light" | "dark") => {
  const event: CustomEventMap["themechange"] = new CustomEvent("themechange", {
    detail: { theme },
  })
  document.dispatchEvent(event)
}

// Emit initial theme
emitThemeChangeEvent("dark")

