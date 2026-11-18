import type {ReactNode} from 'react'

function inline(text: string): ReactNode[] {
  const tokens = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|home page of this site|本站首页)/g
  return text.split(tokens).map((value, key) => {
    if (value.startsWith('**') && value.endsWith('**')) return <strong key={key}>{value.slice(2,-2)}</strong>
    if (value.startsWith('*') && value.endsWith('*')) return <em key={key}>{value.slice(1,-1)}</em>
    if (value.startsWith('`') && value.endsWith('`')) return <code key={key}>{value.slice(1,-1)}</code>
    if (value === 'home page of this site' || value === '本站首页') return <a key={key} href="#/">{value}</a>
    const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(value)
    if (link) return /^https?:\/\//i.test(link[2]) ? <a key={key} href={link[2]}>{link[1]}</a> : link[1]
    return value
  })
}

export function Markdown({source}: {source: string}) {
  const blocks: ReactNode[] = []
  const lines = source.trim().split(/\r?\n/)
  let i = 0
  while (i < lines.length) {
    const key = i
    const line = lines[i++]
    if (!line.trim()) continue
    if (line.startsWith('```')) {
      const body = []
      while (i < lines.length && !lines[i].startsWith('```')) body.push(lines[i++])
      i++
      blocks.push(<pre key={key}><code>{body.join('\n')}</code></pre>)
    } else if (/^#{1,3} /.test(line)) {
      const Heading = (['h1','h2','h3'] as const)[line.indexOf(' ') - 1]
      blocks.push(<Heading key={key}>{inline(line.slice(line.indexOf(' ') + 1))}</Heading>)
    } else if (line.startsWith('|') && /^\|[ :|-]+\|$/.test(lines[i] ?? '')) {
      const cells = (row: string) => row.trim().slice(1,-1).split('|').map(cell => cell.trim())
      const headers = cells(line)
      const rows = []
      i++
      while (i < lines.length && lines[i].startsWith('|')) rows.push(cells(lines[i++]))
      blocks.push(<div className="paper-table" key={key}><table><thead><tr>{headers.map((s,k)=><th scope="col" key={k}>{inline(s)}</th>)}</tr></thead>
        <tbody>{rows.map((row,j)=><tr key={j}>{row.map((s,k)=><td key={k}>{inline(s)}</td>)}</tr>)}</tbody></table></div>)
    } else if (/^(?:- |\d+\. )/.test(line)) {
      const List = line.startsWith('- ') ? 'ul' : 'ol'
      const prefix = List === 'ul' ? /^- / : /^\d+\. /
      const items = [line]
      while (i < lines.length && prefix.test(lines[i])) items.push(lines[i++])
      blocks.push(<List key={key}>{items.map((s,k)=><li key={k}>{inline(s.replace(prefix,''))}</li>)}</List>)
    } else {
      blocks.push(<p key={key}>{inline(line)}</p>)
    }
  }
  return <>{blocks}</>
}
