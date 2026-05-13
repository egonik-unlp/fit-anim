import { makePoly } from "../src/models/poly";
import { makeMLP } from "../src/models/mlp";
import { makeDataset } from "../src/data/presets";
import { step, loss } from "../src/training/loop";

function fit(name: string, model: any, lr: number, steps: number, preset: any) {
  const data = makeDataset(preset, { n: 100, noise: 0.1, seed: 1 });
  let p = model.init(7);
  const l0 = loss(model, data, p);
  for (let i = 0; i < steps; i++) p = step(model, data, p, lr).params;
  const l1 = loss(model, data, p);
  console.log(`${name}: loss ${l0.toExponential(2)} -> ${l1.toExponential(2)} (${(l1 / l0).toExponential(2)}x)`);
  if (!(l1 < l0)) throw new Error(`${name} did not improve`);
}

fit("poly-deg3 / cubic", makePoly(3), 0.05, 400, "cubic");
fit("poly-deg2 / parabola", makePoly(2), 0.1, 300, "parabola");
fit("poly-deg1 / line", makePoly(1), 0.2, 200, "line");
fit("mlp-H16 / sine", makeMLP(16), 0.05, 800, "sine");
fit("mlp-H16 / step", makeMLP(16), 0.05, 800, "step");
console.log("ok");
