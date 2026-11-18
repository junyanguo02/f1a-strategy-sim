// Run against a local production preview and a private Chrome debugging instance.
import {mkdirSync,writeFileSync} from 'node:fs'
const stage=process.argv[2]
if(!['before','after'].includes(stage))throw Error('expected before or after')
const tab=await(await fetch('http://127.0.0.1:9243/json/new?about:blank',{method:'PUT'})).json()
const ws=new WebSocket(tab.webSocketDebuggerUrl),pending=new Map()
await new Promise(resolve=>ws.onopen=resolve)
let seq=0
ws.onmessage=e=>{const m=JSON.parse(e.data);pending.get(m.id)?.(m);pending.delete(m.id)}
async function call(method,params={}){
 const id=++seq,p=new Promise(r=>pending.set(id,r));ws.send(JSON.stringify({id,method,params}))
 const m=await p;if(m.error)throw Error(m.error.message);return m.result
}
async function js(expression){const m=await call('Runtime.evaluate',{expression,returnByValue:true});if(m.exceptionDetails)throw Error(m.result.description);return m.result.value}
async function until(expr){for(let i=0;i<600;i++){if(await js(expr))return;await new Promise(r=>setTimeout(r,150))}throw Error(expr)}
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1250,deviceScaleFactor:1,mobile:false})
await call('Network.enable');await call('Network.setCacheDisabled',{cacheDisabled:true})
await call('Page.navigate',{url:'http://127.0.0.1:4183/f1a-strategy-sim/?capture='+stage+'#/lab'})
await until("!!document.querySelector('.languages [lang=zh]')")
await js("document.querySelector('.languages [lang=zh]').click()")
await until("!!document.querySelector('.pair .panel:nth-child(2) h2')")
const text=await js("document.querySelector('.pair .panel:nth-child(2)').innerText")
console.log(text)
const shot=await call('Page.captureScreenshot',{format:'png'})
mkdirSync('analysis/pitloss_sensitivity/screenshots',{recursive:true})
writeFileSync('analysis/pitloss_sensitivity/screenshots/'+stage+'.png',Buffer.from(shot.data,'base64'))
if(stage==='after'){
 await until("document.body.innerText.includes('当前计划与基准相同')&&document.body.innerText.includes('赛前次选减首选')")
 console.log(await js("JSON.stringify([document.querySelector('.lab aside small').innerText,getComputedStyle(document.querySelector('.handle .num')).color])"))
 await js("document.querySelector('.handle').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))")
 await until("document.querySelectorAll('.pair svg').length===2")
}
ws.close()
