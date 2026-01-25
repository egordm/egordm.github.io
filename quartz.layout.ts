import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { FileTrieNode } from "./quartz/util/fileTrie"

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
    Component.Darkmode(),
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
        { Component: Component.ReaderMode() },
      ],
    }),
    Component.Explorer({
      folderDefaultState: "collapsed",
      sortFn: explorerSortFn,
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
      ],
    }),
    Component.Explorer({
      folderDefaultState: "collapsed",
      sortFn: explorerSortFn,
    }),
  ],
  right: [],
}

export function explorerSortFn(a: FileTrieNode, b: FileTrieNode) {
  const nameA = (a.displayName || "").toLowerCase()
  const nameB = (b.displayName || "").toLowerCase()

  // Sort order: folders first, then files. Sort folders and files alphabetically
  if ((!a.isFolder && !b.isFolder) || (a.isFolder && b.isFolder)) {
    // specific order for top-level folders: projects then blog
    if (nameA === "projects" && nameB === "blog") return -1
    if (nameA === "blog" && nameB === "projects") return 1

    // specific order for blog posts: by date descending
    if (a.slug.startsWith("blog/") && b.slug.startsWith("blog/")) {
      const dateA = a.data?.date ? new Date(a.data.date) : new Date(0)
      const dateB = b.data?.date ? new Date(b.data.date) : new Date(0)
      if (dateA.getTime() !== dateB.getTime()) {
        return dateB.getTime() - dateA.getTime()
      }
    }

    return a.displayName.localeCompare(b.displayName, undefined, {
      numeric: true,
      sensitivity: "base",
    })
  }

  if (!a.isFolder && b.isFolder) {
    return 1
  } else {
    return -1
  }
}
