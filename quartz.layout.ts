import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// Custom sort function for explorer: Home first, then Projects, then Blog (collapsed)
// Within each section, sort alphabetically
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
        filter: (f) => f.slug?.startsWith("blog/") && f.slug !== "blog/index",
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
    // Hide breadcrumbs on home page
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    // Hide article title and meta on home page (the content has its own title)
    Component.ConditionalRender({
      component: Component.ArticleTitle(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.ContentMeta(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.TagList(),
      condition: (page) => page.fileData.slug !== "index",
    }),
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
      title: "Navigate",
      folderDefaultState: "collapsed",  // Keep folders collapsed by default
      folderClickBehavior: "link",
      sortFn: explorerSortFn,
      // Filter out individual blog posts from explorer - just show the Blog folder
      filterFn: (node) => {
        // Always hide tags
        if (node.slugSegment === "tags") return false
        // Hide assets folder
        if (node.slugSegment === "assets") return false
        // Show everything at top level (index, blog, projects folders)
        // For items inside blog folder, only show the folder itself (handled by depth)
        return true
      },
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
      title: "Navigate",
      folderDefaultState: "collapsed",
      folderClickBehavior: "link",
      sortFn: explorerSortFn,
      filterFn: (node) => {
        if (node.slugSegment === "tags") return false
        if (node.slugSegment === "assets") return false
        return true
      },
    }),
  ],
  right: [],
}
