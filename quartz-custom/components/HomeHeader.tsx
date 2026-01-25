import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../../quartz/components/types"
import style from "./styles/homeHeader.scss"

// Custom header for homepage that stacks PageTitle and Bio vertically
export default (() => {
    const HomeHeader: QuartzComponent = ({ cfg, fileData }: QuartzComponentProps) => {
        // Only show on homepage
        if (fileData.slug !== "index") {
            return null
        }

        const title = cfg?.pageTitle ?? "Home"

        return (
            <div className="home-header">
                <h1 className="page-title">
                    <a href="/">{title}</a>
                </h1>
                <p className="bio-description">
                    MLOps engineer passionate about machine learning, energy forecasting, and building high-performance systems. I write about the things I learn.
                </p>
                <div className="bio-social">
                    <a
                        className="social-link"
                        href="https://github.com/egordm/"
                        aria-label="GitHub"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
                        </svg>
                        <span>GitHub</span>
                    </a>
                    <a
                        className="social-link"
                        href="https://www.linkedin.com/in/egor-dmitriev/"
                        aria-label="LinkedIn"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                            <rect x="2" y="9" width="4" height="12"></rect>
                            <circle cx="4" cy="4" r="2"></circle>
                        </svg>
                        <span>LinkedIn</span>
                    </a>
                    <a
                        className="social-link"
                        href="mailto:egordmitriev2@gmail.com"
                        aria-label="Email"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                            <polyline points="22,6 12,13 2,6"></polyline>
                        </svg>
                        <span>Email</span>
                    </a>
                </div>
            </div>
        )
    }

    HomeHeader.css = style
    return HomeHeader
}) satisfies QuartzComponentConstructor
