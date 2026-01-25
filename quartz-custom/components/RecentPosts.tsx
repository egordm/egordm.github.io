import {
    QuartzComponent,
    QuartzComponentConstructor,
    QuartzComponentProps,
} from "../../quartz/components/types"
import { resolveRelative } from "../../quartz/util/path"
import { getDate } from "../../quartz/components/Date"
import { byDateAndAlphabetical } from "../../quartz/components/PageList"
import style from "./styles/recentPosts.scss"

// Custom date formatter
function formatDateShort(d: Date, locale: string = "en-US"): string {
    return d.toLocaleDateString(locale, {
        year: "numeric",
        month: "short",
        day: "numeric",
    })
}

// Truncate description to ~2 lines
function truncateDescription(desc: string, maxLength: number = 150): string {
    if (desc.length <= maxLength) return desc
    return desc.substring(0, maxLength).replace(/\s+\S*$/, '') + '…'
}

interface RecentPostsOptions {
    limit?: number
}

export default ((opts?: RecentPostsOptions) => {
    const limit = opts?.limit ?? 10

    const RecentPosts: QuartzComponent = ({
        cfg,
        fileData,
        allFiles,
    }: QuartzComponentProps) => {
        // Only show on homepage
        if (fileData.slug !== "index") {
            return null
        }

        // Filter only blog posts (exclude index files and drafts)
        const blogPosts = allFiles.filter(
            (f) =>
                f.slug?.startsWith("blog/") &&
                f.slug !== "blog/index" &&
                !f.frontmatter?.draft
        )

        // Sort by date descending
        const sorter = byDateAndAlphabetical(cfg)
        const sortedPosts = blogPosts.sort(sorter).slice(0, limit)

        return (
            <div className="recent-posts">
                <h2>Recent Posts</h2>

                <ul className="post-list">
                    {sortedPosts.map((post) => {
                        const title = post.frontmatter?.title as string
                        const description = (post.frontmatter?.description as string) || post.description || ""
                        const date = getDate(cfg, post)
                        const series = post.frontmatter?.series as string | undefined

                        // Calculate reading time
                        const text = post.text || ""
                        const words = text.trim().split(/\s+/).length
                        const minutes = Math.ceil(words / 200)

                        // Create series link
                        const seriesSlug = series
                            ? `series/${series.toLowerCase().replace(/\s+/g, '-')}`
                            : null

                        return (
                            <li key={post.slug} className="post-item">
                                <a
                                    href={resolveRelative(fileData.slug!, post.slug!)}
                                    className="internal post-title"
                                >
                                    {title}
                                </a>
                                <div className="post-meta">
                                    <span className="post-date">{date && formatDateShort(date, cfg.locale)}</span>
                                    <span className="meta-sep">·</span>
                                    <span className="post-reading-time">{minutes} min read</span>
                                    {seriesSlug && (
                                        <>
                                            <span className="meta-sep">·</span>
                                            <a href={`/${seriesSlug}`} className="post-series-link">
                                                📚 {series}
                                            </a>
                                        </>
                                    )}
                                </div>
                                {description && (
                                    <p className="post-description">{truncateDescription(description)}</p>
                                )}
                            </li>
                        )
                    })}
                </ul>

                <p className="browse-all">
                    <a href="/blog/">→ Browse all posts</a>
                </p>
            </div>
        )
    }

    RecentPosts.css = style
    return RecentPosts
}) satisfies QuartzComponentConstructor
