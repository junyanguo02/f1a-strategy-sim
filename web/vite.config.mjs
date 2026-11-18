import {defineConfig} from 'vite'
import react from '@vitejs/plugin-react'
import {cpSync} from 'node:fs'
import {resolve,basename} from 'node:path'
export default defineConfig({resolve:{alias:{'@articles':resolve(import.meta.dirname,'../docs')}},base:process.env.PITWALL_BASE || '/f1a-strategy-sim/',build:{copyPublicDir:false},plugins:[
 react(),{name:'copy-research-data',writeBundle(options){
 cpSync('public',resolve(options.dir),{recursive:true,filter:path=>!basename(path).startsWith('mock_')})
 }}
]})
