import { write } from "../../quartz/plugins/emitters/helpers"
import { QuartzEmitterPlugin } from "../../quartz/plugins/types"
import { FullSlug } from "../../quartz/util/path"

export const RobotsTxt: QuartzEmitterPlugin = () => {
    return {
        name: "RobotsTxt",
        async *emit(ctx) {
            const cfg = ctx.cfg.configuration
            const base = cfg.baseUrl ?? ""

            const robotsTxtContent = `User-agent: *
Disallow: /BOLD/
Disallow: /BOLDER/
Disallow: /notionsci/
Allow: /

Sitemap: https://${base}/sitemap.xml`

            yield write({
                ctx,
                content: robotsTxtContent,
                slug: "robots" as FullSlug,
                ext: ".txt",
            })
        },
    }
}
