export type Event={race_id:string;year:number;round:number;name:string;has_lab:boolean;has_timeline:boolean}
export type RaceIndex={races:Event[]}
const rounds='巴林 沙特阿拉伯 澳大利亚 日本 中国 迈阿密 艾米利亚－罗马涅 摩纳哥 加拿大 西班牙 奥地利 英国 匈牙利 比利时 荷兰 意大利 阿塞拜疆 新加坡 美国 墨西哥城 圣保罗 拉斯维加斯 卡塔尔 阿布扎比'.split(' ')
export function raceName(r:Event,t:(en:string,zh:string)=>string){
 return r.year+' · '+t(r.name,r.year===2024?rounds[r.round-1]+'大奖赛':r.name)
}
