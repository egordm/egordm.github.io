// @ts-ignore
import darkmodeScript from "./scripts/darkmode.inline"
import styles from "./styles/darkmode.scss"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
// import { i18n } from "../i18n"
// import { classNames } from "../util/lang"

const Darkmode: QuartzComponent = (_props: QuartzComponentProps) => {
  return <></>
}

Darkmode.beforeDOMLoaded = darkmodeScript
Darkmode.css = styles

export default (() => Darkmode) satisfies QuartzComponentConstructor
