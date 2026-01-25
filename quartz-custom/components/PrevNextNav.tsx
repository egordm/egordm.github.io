import {
    QuartzComponentConstructor,
    QuartzComponentProps,
} from "../../quartz/components/types"
import style from "./styles/prevNextNav.scss"

const PrevNextNav: QuartzComponentConstructor = () => {
    const PrevNextNav = ({ fileData, allFiles }: QuartzComponentProps) => {
        // Don't show on index page or non-blog pages
        if (!fileData.slug?.startsWith("blog/") || fileData.slug === "blog/index") {
            return null
        }

        // Get all blog posts sorted by date descending
        const blogPosts = allFiles
            .filter((f) => f.slug?.startsWith("blog/") && f.slug !== "blog/index" && !f.frontmatter?.draft)
            .sort((a, b) => {
                const dateA = new Date(
                    (a.frontmatter?.date as string) || "1970-01-01",
                ).getTime()
                const dateB = new Date(
                    (b.frontmatter?.date as string) || "1970-01-01",
                ).getTime()
                return dateB - dateA // Newest first
            })

        const currentIndex = blogPosts.findIndex((f) => f.slug === fileData.slug)
        if (currentIndex === -1) return null

        // In DESC order: previous is index-1 (newer), next is index+1 (older)
        const newer = blogPosts[currentIndex - 1]
        const older = blogPosts[currentIndex + 1]

        if (!newer && !older) return null

        return (
            <nav className="prev-next-nav">
                <div className="prev-next-links">
                    <div className="nav-link nav-prev">
                        {older && (
                            <a href={`/${older.slug}`} rel="prev">
                                <span className="nav-direction">← Older</span>
                                <span className="nav-title">{older.frontmatter?.title}</span>
                            </a>
                        )}
                    </div>
                    <div className="nav-link nav-next">
                        {newer && (
                            <a href={`/${newer.slug}`} rel="next">
                                <span className="nav-direction">Newer →</span>
                                <span className="nav-title">{newer.frontmatter?.title}</span>
                            </a>
                        )}
                    </div>
                </div>
            </nav>
        )
    }

    PrevNextNav.css = style
    return PrevNextNav
}

export default PrevNextNav
