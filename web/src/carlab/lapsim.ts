// Same segment-midpoint force balance and circular relaxation as f1a.lapsim.
export interface Car {
  mass_kg: number; power_kW: number; mu: number; cla_m2: number; cda_m2: number
  brake_max_mps2: number; v_max_mps: number; front_share: number
  static_front_share: number; rho_kg_m3: number
}
export interface Wake { gap: number; a: number; b: number; g0: number }
export function applyWake(car: Car, wake?: Wake): Car {
  if (!wake) return car
  const {a,b,gap,g0}=wake
  if (!(gap>=0 && g0>0 && Number.isFinite(g0) && a>=0 && a<1 && b>=0 && b<1)) throw Error('Invalid wake parameters')
  const exposure=Math.exp(-gap/g0)
  return {...car,cla_m2:car.cla_m2*(1-a*exposure),mu:car.mu*(1-b*exposure)}
}
const clip=(v:number,lo:number,hi:number)=>Math.max(lo,Math.min(hi,v))

export function solveLap(s: number[], curvature: number[], car: Car) {
  const n=curvature.length
  if(n<3 || (s.length!==n && s.length!==n+1)) throw Error('Invalid grid dimensions')
  const load=.5*car.rho_kg_m3*car.cla_m2/car.mass_kg
  const drag=.5*car.rho_kg_m3*car.cda_m2/car.mass_kg
  const lateral=load*Math.min(car.front_share/car.static_front_share,(1-car.front_share)/(1-car.static_front_share))
  const power=1000*car.power_kW/car.mass_kg
  const top=Math.min(car.v_max_mps,drag?Math.cbrt(power/drag):Infinity)
  let q=new Float64Array(n), next=new Float64Array(n)
  const ds=new Float64Array(n),km=new Float64Array(n),eq=new Float64Array(n)
  for(let i=0;i<n;i++){
    ds[i]=i+1<s.length?s[i+1]-s[i]:s[1]-s[0]
    if(!(ds[i]>0 && Number.isFinite(ds[i]) && Number.isFinite(curvature[i]))) throw Error('Invalid geometry')
    if(s.length===n && Math.abs(ds[i]-ds[0])>1e-9+1e-7*ds[0]) throw Error('Nonuniform grid requires edges')
    km[i]=Math.abs((curvature[i]+curvature[(i+1)%n])*.5)
    const denom=Math.abs(curvature[i])-car.mu*lateral
    q[i]=Math.min(top**2,denom>0?car.mu*9.81/denom:Infinity)
  }
  const cap=q.reduce((max,v)=>Math.max(max,v),0)
  if(!Number.isFinite(cap)) throw Error('Unbounded speed envelope')
  for(let i=0;i<n;i++){
    let lo=0,hi=cap
    for(let j=0;j<48;j++){
      const m=(lo+hi)*.5
      if((km[i]*m/(car.mu*(9.81+lateral*m)))**2+(drag*m/(car.mu*(9.81+load*m)))**2<=1) lo=m
      else hi=m
    }
    eq[i]=(lo+hi)*.5
  }
  function grip(i:number,x:number){
    const used=Math.min(1,km[i]*x/(car.mu*(9.81+lateral*x)))
    return car.mu*(9.81+load*x)*Math.sqrt((1-used)*(1+used))
  }
  const drive=(i:number,x:number)=>Math.min(grip(i,x),power/Math.sqrt(Math.max(x,1e-15)))-drag*x
  const brake=(i:number,x:number)=>Math.min(car.brake_max_mps2,grip(i,x))+drag*x
  for(let iteration=0;iteration<2*n+64;iteration++){
    next.set(q)
    for(let i=0;i<n;i++){
      const j=(i+1)%n, start=q[i],end=q[j],lo=Math.min(start,eq[i]),hi=Math.max(start,eq[i])
      const mid=clip(start+ds[i]*drive(i,start),lo,hi)
      const driven=mid===eq[i]?mid:clip(start+2*ds[i]*drive(i,mid),lo,hi)
      next[j]=Math.min(next[j],driven)
      next[i]=Math.min(next[i],end+2*ds[i]*brake(i,end+ds[i]*brake(i,end)))
    }
    let change=0
    for(let i=0;i<n;i++){
      if(!(next[i]>0)) throw Error('Invalid speed envelope')
      change=Math.max(change,q[i]-next[i])
    }
    const tmp=q;q=next;next=tmp
    if(change<1e-8){
      const v=q.map(Math.sqrt),t=new Float64Array(n+1)
      for(let i=0;i<n;i++) t[i+1]=t[i]+2*ds[i]/(v[i]+v[(i+1)%n])
      return {v,t,total:t[n]}
    }
  }
  throw Error('Periodic speed constraints did not converge')
}
