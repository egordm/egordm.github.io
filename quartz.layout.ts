import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// Custom sort function for explorer: Home first, then Projects, then Blog
const explorerSortFn = (a: any, b: any) => {
  // Define priority order for top-level items
  const priority: Record<string, number> = {
    "index": 0,      // Home
    "projects": 1,   // Projects folder
    "blog": 2,       // Blog folder (last since it has many items)
  }

  // Get the segment name (folder or file name)
  const aName = a.slugSegment || a.displayName?.toLowerCase() || ""
  const bName = b.slugSegment || b.displayName?.toLowerCase() || ""

  const aPriority = priority[aName] ?? 99
  const bPriority = priority[bName] ?? 99

  if (aPriority !== bPriority) {
    return aPriority - bPriority
  }

  // If same priority, folders first, then alphabetical
  if (a.isFolder !== b.isFolder) {
    return a.isFolder ? -1 : 1
  }

  return a.displayName.localeCompare(b.displayName, undefined, {
    numeric: true,
    sensitivity: "base",
  })
}

// Filter function for explorer - hide individual blog posts, only show main sections
const explorerFilterFn = (node: any) => {
  const slug = node.slug as string

  // Always hide tags
  if (node.slugSegment === "tags") return false
  // Hide assets folders
  if (node.slugSegment === "assets") return false

  // Hide individual blog posts - only show the Blog folder itself
  // Blog folder has slug "blog/index", individual posts have slug "blog/post-name"
  if (slug && slug.startsWith("blog/") && slug !== "blog/index") {
    return false
  }

  return true
}

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [
    // Show recent notes only on the home page
    Component.ConditionalRender({
      component: Component.RecentNotes({
        title: "Recent Blog Posts",
        limit: 5,
        linkToMore: "blog/" as any,
        showTags: true,
        filter: (f) => (f.slug ?? "").startsWith("blog/") && f.slug !== "blog/index",
      }),
      condition: (page) => page.fileData.slug === "index",
    }),
  ],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/egordm/",
      LinkedIn: "https://www.linkedin.com/in/egor-dmitriev/",
      "RSS Feed": "/index.xml",
    },
  }),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
        { Component: Component.ReaderMode() },
      ],
    }),
    Component.Explorer({
      sortFn: explorerSortFn,
      // filterFn: explorerFilterFn,
    }),
  ],
  right: [
    Component.Graph(),
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Backlinks(),
  ],
}

// components for pages that display lists of pages (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        {
          Component: Component.Search(),
          grow: true,
        },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer({
      sortFn: explorerSortFn,
      // filterFn: explorerFilterFn,
    }),
  ],
  right: [],
}
