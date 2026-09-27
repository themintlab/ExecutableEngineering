import json

path = '/home/wellandm/Code/ExecutableEngineering/chapters/interpolation_and_curve_fitting/interpolation/interpolation.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

# Combine 1, 2, 3 concepts into one slide (Cell 2, 3, 4)
nb['cells'][2]['source'] = [
    "## Introduction to Interpolation\n",
    "\n",
    "Interpolation is a fundamental numerical method for fitting continuous curves to discrete data points. Historically, it was easier to perform locally or iteratively as new information was obtained."
]

nb['cells'][3]['source'] = [
    "When choosing an interpolation method, we must consider:\n",
    "* **Build Speed**: How fast can we construct the model?\n",
    "* **Update Speed**: How quickly can we add new data?\n",
    "* **Execution Speed**: How fast can we evaluate interpolated values?\n",
    "* **Generalizability**: Does it scale to N-dimensions?"
]

nb['cells'][4]['source'] = [
    "The polynomial methods discussed here rely on a core property of linear algebra:\n",
    "\n",
    "**It is always possible to construct a *unique* polynomial of degree $n$ that passes exactly through $n + 1$ distinct data points.**"
]

# Slide 4 concepts (Cell 5, 6, 7)
nb['cells'][5]['source'] = [
    "## Example: Interpolating a Gaussian Curve\n",
    "\n",
    "For illustrative purposes, let's design a toy problem for exploration. We will sample a known Gaussian function, and then attempt to recover it using various interpolation techniques."
]

nb['cells'][7]['source'] = [
    "Our goal is to use the sampled data (the red points) and recover the 'true' underlying function (the blue curve) as faithfully as possible."
]

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Re-chunked interpolation.ipynb")
