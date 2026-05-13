import type { Model, Params } from "./types";

export function makePoly(degree: number): Model {
  const n = degree + 1;
  return {
    id: `poly-${degree}`,
    label: `Polynomial (degree ${degree})`,
    init() {
      return new Array(n).fill(0);
    },
    forward(x, params) {
      let y = 0;
      let xk = 1;
      for (let k = 0; k < n; k++) {
        y += params[k]! * xk;
        xk *= x;
      }
      return y;
    },
    gradient(x, y, params) {
      let yhat = 0;
      let xk = 1;
      const powers = new Array(n);
      for (let k = 0; k < n; k++) {
        powers[k] = xk;
        yhat += params[k]! * xk;
        xk *= x;
      }
      const err = yhat - y;
      const g = new Array(n);
      for (let k = 0; k < n; k++) g[k] = 2 * err * powers[k];
      return g;
    },
  };
}
