import type { Model, Params } from "./types";
import { mulberry32, gaussian } from "./rng";

// 1 hidden layer MLP: x -> H tanh units -> 1 linear output.
// Params layout (length 3H + 1):
//   [0 .. H-1]       w1 (input -> hidden)
//   [H .. 2H-1]      b1 (hidden bias)
//   [2H .. 3H-1]     w2 (hidden -> output)
//   [3H]             b2 (output bias)
export function makeMLP(hidden: number): Model {
  const H = hidden;
  const N = 3 * H + 1;

  function unpack(p: Params) {
    return {
      w1: p.slice(0, H),
      b1: p.slice(H, 2 * H),
      w2: p.slice(2 * H, 3 * H),
      b2: p[3 * H]!,
    };
  }

  return {
    id: `mlp-${H}`,
    label: `MLP (1 hidden, H=${H})`,
    init(seed) {
      const rand = mulberry32(seed);
      const p = new Array(N).fill(0);
      // Glorot-ish init for tanh
      const s = Math.sqrt(1 / Math.max(1, H));
      for (let i = 0; i < H; i++) {
        p[i] = gaussian(rand) * 1.0;          // w1
        p[H + i] = gaussian(rand) * 0.5;      // b1
        p[2 * H + i] = gaussian(rand) * s;    // w2
      }
      p[3 * H] = 0; // b2
      return p;
    },
    forward(x, params) {
      const { w1, b1, w2, b2 } = unpack(params);
      let y = b2;
      for (let i = 0; i < H; i++) {
        y += w2[i]! * Math.tanh(w1[i]! * x + b1[i]!);
      }
      return y;
    },
    gradient(x, y, params) {
      const { w1, b1, w2, b2 } = unpack(params);
      // forward
      const h = new Array<number>(H);
      let yhat = b2;
      for (let i = 0; i < H; i++) {
        h[i] = Math.tanh(w1[i]! * x + b1[i]!);
        yhat += w2[i]! * h[i]!;
      }
      const err = yhat - y;
      const dLdy = 2 * err;
      const g = new Array<number>(N).fill(0);
      // dL/dw2_i = dLdy * h_i ; dL/db2 = dLdy
      for (let i = 0; i < H; i++) {
        g[2 * H + i] = dLdy * h[i]!;
        const dLdh = dLdy * w2[i]!;
        const dhdz = 1 - h[i]! * h[i]!; // tanh'
        const dLdz = dLdh * dhdz;
        g[i] = dLdz * x;       // w1
        g[H + i] = dLdz;       // b1
      }
      g[3 * H] = dLdy;
      return g;
    },
  };
}
