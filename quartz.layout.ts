import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import * as CustomComponent from "./quartz-custom/components"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [
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
    // Show HomeHeader only on homepage (includes title + bio)
    CustomComponent.HomeHeader(),
    Component.ConditionalRender({
      component: Component.Search(),
      condition: (page) => page.fileData.slug === "index",
    }),
    // Show breadcrumbs on content pages
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    // Show article title on content pages
    Component.ConditionalRender({
      component: Component.ArticleTitle(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    // Show content meta on content pages
    Component.ConditionalRender({
      component: Component.ContentMeta(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.TagList(),
    // Show series banner for blog posts that are part of a series
    CustomComponent.SeriesBanner(),
  ],
  left: [
    Component.ConditionalRender({
      component: Component.PageTitle(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.MobileOnly(Component.Spacer()),
    Component.ConditionalRender({
      component: Component.Search(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.DesktopOnly(
        CustomComponent.SidebarPosts({
          title: "Recent Posts",
          limit: 4,
          filter: (f) => f.slug!.startsWith("blog/") && f.slug! !== "blog/index" && !f.frontmatter?.draft,
          linkToMore: "blog/" as any,
        }),
      ),
      condition: (page) => page.fileData.slug !== "index",
    }),
  ],
  right: [
    // Hide TOC on homepage
    Component.ConditionalRender({
      component: Component.DesktopOnly(Component.TableOfContents()),
      condition: (page) => page.fileData.slug !== "index",
    }),
  ],
  afterBody: [
    // Show recent posts on homepage (automated)
    CustomComponent.RecentPosts({ limit: 15 }),
    // Show prev/next navigation for blog posts
    CustomComponent.PrevNextNav(),
  ],
}

// components for pages that display lists of pages (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.ConditionalRender({
      component: Component.PageTitle(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.MobileOnly(Component.Spacer()),
    Component.ConditionalRender({
      component: Component.Search(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ConditionalRender({
      component: Component.DesktopOnly(
        CustomComponent.SidebarPosts({
          title: "Recent Posts",
          limit: 4,
          filter: (f) => f.slug!.startsWith("blog/") && f.slug! !== "blog/index" && !f.frontmatter?.draft,
          linkToMore: "blog/" as any,
        }),
      ),
      condition: (page) => page.fileData.slug !== "index",
    }),
  ],
  right: [],
}
