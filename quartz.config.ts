import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"
import * as CustomPlugin from "./quartz-custom/plugins"

/**
 * Quartz 4 Configuration
 * EgorDm's Personal Website
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Egor Dmitriev",
    pageTitleSuffix: " | egordmitriev.dev",
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    locale: "en-US",
    baseUrl: "egordmitriev.dev",
    ignorePatterns: ["private", "templates", ".obsidian", "_archive_zola"],
    defaultDateType: "created",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Inter",
        body: "Inter",
        code: "JetBrains Mono",
      },
      colors: {
        // Obsidian-inspired dark theme with purple accents
        lightMode: {
          light: "#ffffff",
          lightgray: "#f2f2f2",
          gray: "#b8b8b8",
          darkgray: "#4e4e4e",
          dark: "#1a1a1a",
          secondary: "#7c3aed",
          tertiary: "#a78bfa",
          highlight: "rgba(124, 58, 237, 0.1)",
          textHighlight: "#fef08a88",
        },
        darkMode: {
          light: "#1e1e1e",
          lightgray: "#2d2d2d",
          gray: "#5c5c5c",
          darkgray: "#d4d4d4",
          dark: "#ebebec",
          secondary: "#a78bfa",
          tertiary: "#7c3aed",
          highlight: "rgba(167, 139, 250, 0.15)",
          textHighlight: "#a78bfa33",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "one-dark-pro",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      Plugin.CNAME(),
      // Comment out CustomOgImages to speed up build time
      // Plugin.CustomOgImages(),
      CustomPlugin.RobotsTxt()
    ],
  },
}

export default config
