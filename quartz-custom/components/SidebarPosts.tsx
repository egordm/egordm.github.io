import {
    QuartzComponent,
    QuartzComponentConstructor,
    QuartzComponentProps,
} from "../../quartz/components/types"
import { SimpleSlug, resolveRelative } from "../../quartz/util/path"
import { QuartzPluginData } from "../../quartz/plugins/vfile"
import { byDateAndAlphabetical } from "../../quartz/components/PageList"
import { Date, getDate } from "../../quartz/components/Date"
import { GlobalConfiguration } from "../../quartz/cfg"
import { classNames } from "../../quartz/util/lang"
import style from "./styles/sidebarPosts.scss"

interface Options {
    title?: string
    limit: number
    linkToMore: SimpleSlug | false
    filter: (f: QuartzPluginData) => boolean
    sort: (f1: QuartzPluginData, f2: QuartzPluginData) => number
}

const defaultOptions = (cfg: GlobalConfiguration): Options => ({
    limit: 4,
    linkToMore: false,
    filter: () => true,
    sort: byDateAndAlphabetical(cfg),
})

export default ((userOpts?: Partial<Options>) => {
    const SidebarPosts: QuartzComponent = ({
        allFiles,
        fileData,
        displayClass,
        cfg,
    }: QuartzComponentProps) => {
        const opts = { ...defaultOptions(cfg), ...userOpts }
        const pages = allFiles.filter(opts.filter).sort(opts.sort)
        const remaining = Math.max(0, pages.length - opts.limit)

        return (
            <div class={classNames(displayClass, "sidebar-posts")}>
                <h3>{opts.title ?? "Recent Posts"}</h3>
                <ul class="sidebar-posts-ul">
                    {pages.slice(0, opts.limit).map((page) => {
                        const title = page.frontmatter?.title ?? "Untitled"
                        const series = page.frontmatter?.series as string | undefined

                        return (
                            <li class="sidebar-post-li">
                                <div class="sidebar-post-header">
                                    <a
                                        href={resolveRelative(fileData.slug!, page.slug!)}
                                        class="internal sidebar-post-title"
                                    >
                                        {title}
                                    </a>
                                </div>
                                <div class="sidebar-post-meta">
                                    {series && (
                                        <span class="sidebar-series-badge">{series}</span>
                                    )}
                                    {page.dates && (
                                        <span class="sidebar-post-date">
                                            <Date date={getDate(cfg, page)!} locale={cfg.locale} />
                                        </span>
                                    )}
                                </div>
                            </li>
                        )
                    })}
                </ul>
                {opts.linkToMore && remaining > 0 && (
                    <p class="sidebar-browse">
                        <a href={resolveRelative(fileData.slug!, opts.linkToMore)}>
                            All posts →
                        </a>
                    </p>
                )}
            </div>
        )
    }

    SidebarPosts.css = style
    return SidebarPosts
}) satisfies QuartzComponentConstructor
