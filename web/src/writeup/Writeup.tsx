import {useContext,useEffect} from 'react'
import {Language,useText} from '../locale'
import {Markdown} from './Markdown'
import en from '@articles/writeup_en.md?raw'
import zh from '@articles/writeup_zh.md?raw'
import enURL from '@articles/writeup_en.md?url'
import zhURL from '@articles/writeup_zh.md?url'

export default function Writeup() {
  const lang = useContext(Language)
  const t = useText()
  const source = lang === 'en' ? en : zh
  useEffect(() => {
    const defaultTitle = document.title
    const description = document.createElement('meta')
    description.name = 'description'
    description.content = source.split('\n\n')[3].trim()
    document.title = source.split('\n')[0].slice(2) + ' · PITWALL'
    document.head.append(description)
    return () => { document.title = defaultTitle; description.remove() }
  }, [source])
  return <article className="paper" lang={lang === 'en' ? 'en' : 'zh-CN'}>
    <Markdown source={source}/>
    <div className="paper-links">
      <a href={lang === 'en' ? enURL : zhURL} download={'writeup_'+lang+'.md'}>{t('Download as Markdown','下载 Markdown 原文')}</a>
      <a href="https://github.com/junyanguo02/f1a-strategy-sim">{t('GitHub repository','GitHub 仓库')}</a>
    </div>
  </article>
}
