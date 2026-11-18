/** Mulberry32 with open-interval uniforms, Box–Muller normals and
 * Marsaglia–Tsang gamma sampling (including the shape < 1 transformation). */
export class Random {
  private spare: number | undefined
  constructor(private state: number) {}

  uniform(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0
    let n = Math.imul(this.state ^ (this.state >>> 15), this.state | 1)
    n ^= n + Math.imul(n ^ (n >>> 7), n | 61)
    return (((n ^ (n >>> 14)) >>> 0) + 0.5) / 4294967296
  }

  normal(): number {
    if (this.spare !== undefined) {
      const value = this.spare
      this.spare = undefined
      return value
    }
    const r = Math.sqrt(-2 * Math.log(this.uniform()))
    const angle = 2 * Math.PI * this.uniform()
    this.spare = r * Math.sin(angle)
    return r * Math.cos(angle)
  }

  gamma(shape: number): number {
    if (shape < 1) return this.gamma(shape + 1) * this.uniform() ** (1 / shape)
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d)
    for (;;) {
      const x = this.normal(), root = 1 + c * x
      if (root <= 0) continue
      const v = root ** 3, u = this.uniform()
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
    }
  }

  student(nu: number): number {
    return this.normal() / Math.sqrt(2 * this.gamma(nu / 2) / nu)
  }

  shuffle(a: number[]) {
    for (let i = a.length - 1; i > 0; --i) {
      const j = Math.floor(this.uniform() * (i + 1))
      ;[a[i], a[j]] = [a[j], a[i]]
    }
  }
}
