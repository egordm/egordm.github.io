import {
    QuartzComponentConstructor,
    QuartzComponentProps,
} from "../../quartz/components/types"
import style from "./styles/seriesBanner.scss"

interface SeriesPost {
    slug: string
    title: string
    order: number
}

const SeriesBanner: QuartzComponentConstructor = () => {
    const SeriesBanner = ({ fileData, allFiles }: QuartzComponentProps) => {
        const series = fileData.frontmatter?.series as string | undefined
        const seriesOrder = fileData.frontmatter?.series_order as number | undefined

        // Only show if this post is part of a series
        if (!series || !seriesOrder) {
            return null
        }

        // Find all posts in this series
        const seriesPosts: SeriesPost[] = allFiles
            .filter((f) => f.frontmatter?.series === series && f.frontmatter?.series_order)
            .map((f) => ({
                slug: f.slug || "",
                title: f.frontmatter?.title as string || "Untitled",
                order: f.frontmatter?.series_order as number,
            }))
            .sort((a, b) => a.order - b.order)

        const totalParts = seriesPosts.length

        // Create series slug from series name
        const seriesSlug = `series/${series.toLowerCase().replace(/\s+/g, '-')}`

        return (
            <div className="series-banner">
                <details className="series-toc">
                    <summary>
                        <span className="series-icon">📚</span>
                        <a href={`/${seriesSlug}`} className="series-name">{series}</a>
                        <span className="series-position">Part {seriesOrder} of {totalParts}</span>
                    </summary>
                    <ol className="series-list">
                        {seriesPosts.map((post) => (
                            <li
                                key={post.slug}
                                className={post.slug === fileData.slug ? "current" : ""}
                            >
                                {post.slug === fileData.slug ? (
                                    <span className="current-post">{post.title}</span>
                                ) : (
                                    <a href={`/${post.slug}`}>{post.title}</a>
                                )}
                            </li>
                        ))}
                    </ol>
                </details>
            </div>
        )
    }

    SeriesBanner.css = style
    return SeriesBanner
}

export default SeriesBanner
