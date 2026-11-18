// Run from web/ with the installed esbuild binary (see audit.py).
import fs from 'node:fs'
import {createHash} from 'node:crypto'
import {commonDraws,simulate,pitPicker} from '../src/sim/engine'
import {Random} from '../src/sim/random'
import {unpack,asStrategy} from '../src/sim/inputs'

for(const round of [1,3,10]){
 const folder='public/data/races/2024-'+round+'/'
 const hashes=['sim_inputs_prerace','strategies'].map(n=>createHash('sha256').update(fs.readFileSync(folder+n+'.json')).digest('hex'))
 const doc=JSON.parse(fs.readFileSync(folder+'sim_inputs_prerace.json','utf8'))
 const candidates=JSON.parse(fs.readFileSync(folder+'strategies.json','utf8')).candidates.slice(0,3)
 const plans=candidates.map(asStrategy)
 const {race,tyre}=unpack(doc), means=[]
 const pick=pitPicker(race),pitMeans:number[][]=[]
 const samples:Record<string,number[]>={noise:[],nu:[],stint:[],SC:[],VSC:[],uniform:[],normal:[]}
 const starts=Array.from({length:2},()=>Array(race.total_laps).fill(0))
 const rnd=new Random(39039)
 for(let k=0;k<120000;k++){samples.uniform.push(rnd.uniform());samples.normal.push(rnd.normal())}
 for(let seed=0;seed<100;seed++){
   const D=commonDraws(race,tyre,5000,390000+seed)
   const [a,b,c]=plans.map((p:ReturnType<typeof asStrategy>)=>simulate(race,tyre,p,D))
   means.push([0,b.reduce((sum,v,i)=>sum+(v-a[i]),0)/5000,c.reduce((sum,v,i)=>sum+(v-a[i]),0)/5000])
   const loss=plans.map((plan:ReturnType<typeof asStrategy>)=>{
     let sum=0
     for(const [lap] of plan.stops)for(let i=0;i<D.n_sims;i++)sum+=pick(D.status[i][lap-1],D.phase[i][lap-1],D.pit_u[i][lap-1])
     return sum/D.n_sims
   })
   pitMeans.push(loss.map((v:number)=>v-loss[0]))
   for(let i=0;i<D.n_sims;i++){
     if(samples.noise.length<120000){
       samples.noise.push(D.t_noise[i][0]);samples.nu.push(tyre.nu[D.d[i]])
       samples.stint.push(D.u_stint[i][0]*tyre.sd_stint[D.d[i]])
     }
     for(let j=0;j<race.total_laps;j++)if(D.phase[i][j]===1){
       const code=D.status[i][j],key=code===2?'SC':'VSC';starts[code-1][j]++
       let end=j+1;while(end<race.total_laps&&D.phase[i][end]>1)end++
       if(samples[key].length<120000)samples[key].push(end-j)
     }
   }
 }
 fs.writeFileSync('.qa039/ts'+round+'.json',JSON.stringify({means,starts,samples,hashes,pitMeans}))
}
