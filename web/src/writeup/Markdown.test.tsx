import {renderToStaticMarkup} from 'react-dom/server'
import {expect,it} from 'vitest'
import {Markdown} from './Markdown'

const render = (source: string) => renderToStaticMarkup(<Markdown source={source}/>)
it('renders column headers and formatted cells', () => {
  const html = render('| Mode | Less |\n| --- | --- |\n| Clean | **79.9%** |')
  expect(html).toContain('<th scope="col">Mode</th>')
  expect(html).toContain('<td><strong>79.9%</strong></td>')
})
it('preserves literal fenced code and escapes markup', () => {
  expect(render('```\n<a> * b\n```')).toContain('<pre><code>&lt;a&gt; * b</code></pre>')
})
it('links the home phrases and permits only HTTP(S) Markdown links', () => {
  const html = render('本站首页 home page of this site [GitHub](https://github.com) [unsafe](javascript:alert)')
  expect(html.match(/href="#\/"/g)).toHaveLength(2)
  expect(html).toContain('href="https://github.com"')
  expect(html).not.toContain('javascript:')
})
