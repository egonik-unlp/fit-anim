export type Params = number[];

export interface Model {
  readonly id: string;
  readonly label: string;
  init(seed: number): Params;
  forward(x: number, params: Params): number;
  /** dL/dp for each param at one (x, y) sample. Loss is squared error (yhat - y)^2. */
  gradient(x: number, y: number, params: Params): Params;
}
