(() => {
  "use strict";

  const svgNamespace = "http://www.w3.org/2000/svg";
  const byId = (id) => document.getElementById(id);
  const formatNumber = (value, digits = 2) =>
    Number(value).toLocaleString("pt-BR", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  const sigmoid = (value) => 1 / (1 + Math.exp(-value));

  function addSvgElement(svg, tag, attributes = {}, text) {
    const element = document.createElementNS(svgNamespace, tag);
    Object.entries(attributes).forEach(([name, value]) =>
      element.setAttribute(name, value),
    );
    if (text !== undefined) element.textContent = text;
    svg.appendChild(element);
    return element;
  }

  function addLine(svg, x1, y1, x2, y2, className) {
    return addSvgElement(svg, "line", {
      x1,
      y1,
      x2,
      y2,
      class: className,
    });
  }

  function drawLogitLab() {
    const x = Number(byId("logit-x").value);
    const beta = Number(byId("logit-beta").value);
    const beta0 = Number(byId("logit-beta0").value);
    const eta = beta0 + beta * x;
    const probability = sigmoid(eta);
    const odds = probability / (1 - probability);

    byId("logit-x-value").textContent = formatNumber(x, 1);
    byId("logit-beta-value").textContent = formatNumber(beta, 1);
    byId("logit-beta0-value").textContent = formatNumber(beta0, 1);
    byId("logit-eta").textContent = formatNumber(eta);
    byId("logit-prob").textContent = formatNumber(probability);
    byId("logit-odds").textContent = formatNumber(odds);

    if (beta > 0) {
      byId("logit-interpretation").textContent =
        "β₁ positivo: aumentar x eleva a probabilidade estimada.";
    } else if (beta < 0) {
      byId("logit-interpretation").textContent =
        "β₁ negativo: aumentar x reduz a probabilidade estimada.";
    } else {
      byId("logit-interpretation").textContent =
        "β₁ igual a zero: x não altera a probabilidade estimada.";
    }

    const svg = byId("logit-chart");
    svg.replaceChildren();
    const left = 70;
    const right = 755;
    const top = 24;
    const bottom = 270;
    const scaleX = (value) => left + ((value + 3) / 6) * (right - left);
    const scaleY = (value) => bottom - value * (bottom - top);

    [0, 0.25, 0.5, 0.75, 1].forEach((value) => {
      addLine(svg, left, scaleY(value), right, scaleY(value), "grid-line");
      addSvgElement(
        svg,
        "text",
        { x: left - 12, y: scaleY(value) + 4, "text-anchor": "end" },
        formatNumber(value, 2),
      );
    });
    [-3, -2, -1, 0, 1, 2, 3].forEach((value) =>
      addSvgElement(
        svg,
        "text",
        { x: scaleX(value), y: bottom + 24, "text-anchor": "middle" },
        String(value),
      ),
    );
    addLine(svg, left, top, left, bottom, "axis");
    addLine(svg, left, bottom, right, bottom, "axis");
    addSvgElement(
      svg,
      "text",
      { x: (left + right) / 2, y: 320, "text-anchor": "middle" },
      "atributo padronizado x",
    );
    addSvgElement(
      svg,
      "text",
      {
        x: 18,
        y: (top + bottom) / 2,
        transform: `rotate(-90 18 ${(top + bottom) / 2})`,
        "text-anchor": "middle",
      },
      "probabilidade estimada",
    );

    let path = "";
    for (let index = 0; index <= 180; index += 1) {
      const currentX = -3 + (6 * index) / 180;
      const currentProbability = sigmoid(beta0 + beta * currentX);
      path += `${index === 0 ? "M" : "L"}${scaleX(currentX).toFixed(2)},${scaleY(currentProbability).toFixed(2)} `;
    }
    addSvgElement(svg, "path", { d: path, class: "main-curve" });
    addLine(svg, scaleX(x), bottom, scaleX(x), scaleY(probability), "guide-line");
    addSvgElement(svg, "circle", {
      cx: scaleX(x),
      cy: scaleY(probability),
      r: 9,
      class: "active-point",
    });
    addSvgElement(
      svg,
      "text",
      { x: scaleX(x) + 14, y: scaleY(probability) - 12 },
      `π̂ = ${formatNumber(probability)}`,
    );
  }

  const thresholdCases = [
    { probability: 0.08, observed: 0 },
    { probability: 0.14, observed: 0 },
    { probability: 0.22, observed: 1 },
    { probability: 0.31, observed: 0 },
    { probability: 0.37, observed: 0 },
    { probability: 0.44, observed: 1 },
    { probability: 0.53, observed: 0 },
    { probability: 0.58, observed: 1 },
    { probability: 0.64, observed: 1 },
    { probability: 0.71, observed: 0 },
    { probability: 0.79, observed: 1 },
    { probability: 0.86, observed: 1 },
    { probability: 0.93, observed: 1 },
  ];

  function drawThresholdLab() {
    const threshold = Number(byId("threshold-value").value);
    byId("threshold-label").textContent = formatNumber(threshold);

    let truePositive = 0;
    let trueNegative = 0;
    let falsePositive = 0;
    let falseNegative = 0;
    thresholdCases.forEach(({ probability, observed }) => {
      const predicted = probability >= threshold ? 1 : 0;
      if (observed === 1 && predicted === 1) truePositive += 1;
      if (observed === 0 && predicted === 0) trueNegative += 1;
      if (observed === 0 && predicted === 1) falsePositive += 1;
      if (observed === 1 && predicted === 0) falseNegative += 1;
    });

    const precision = truePositive / Math.max(1, truePositive + falsePositive);
    const recall = truePositive / Math.max(1, truePositive + falseNegative);
    const accuracy =
      (truePositive + trueNegative) / thresholdCases.length;

    byId("threshold-tp").textContent = truePositive;
    byId("threshold-tn").textContent = trueNegative;
    byId("threshold-fp").textContent = falsePositive;
    byId("threshold-fn").textContent = falseNegative;
    byId("threshold-precision").textContent = formatNumber(precision);
    byId("threshold-recall").textContent = formatNumber(recall);
    byId("threshold-accuracy").textContent = formatNumber(accuracy);

    if (threshold < 0.5) {
      byId("threshold-interpretation").textContent =
        "Limiar menor: mais casos são classificados como 1. A revocação tende a subir, mas podem surgir falsos positivos.";
    } else if (threshold > 0.5) {
      byId("threshold-interpretation").textContent =
        "Limiar maior: a classe 1 exige mais evidência. Falsos positivos tendem a cair, mas podem surgir falsos negativos.";
    } else {
      byId("threshold-interpretation").textContent =
        "O limiar 0,50 é uma escolha possível, não uma consequência obrigatória do modelo.";
    }

    const svg = byId("threshold-chart");
    svg.replaceChildren();
    const left = 72;
    const right = 755;
    const top = 30;
    const bottom = 235;
    const scaleX = (value) => left + value * (right - left);

    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((value) => {
      addLine(svg, scaleX(value), top, scaleX(value), bottom, "grid-line");
      addSvgElement(
        svg,
        "text",
        { x: scaleX(value), y: bottom + 24, "text-anchor": "middle" },
        formatNumber(value, 1),
      );
    });
    addLine(svg, left, bottom, right, bottom, "axis");
    addSvgElement(
      svg,
      "text",
      { x: (left + right) / 2, y: 288, "text-anchor": "middle" },
      "probabilidade estimada π̂",
    );
    addLine(
      svg,
      scaleX(threshold),
      top,
      scaleX(threshold),
      bottom,
      "guide-line",
    );
    addSvgElement(
      svg,
      "text",
      { x: scaleX(threshold), y: 19, "text-anchor": "middle" },
      `c = ${formatNumber(threshold)}`,
    );

    thresholdCases.forEach(({ probability, observed }) => {
      const predicted = probability >= threshold ? 1 : 0;
      const incorrect = predicted !== observed;
      addSvgElement(svg, "circle", {
        cx: scaleX(probability),
        cy: observed === 1 ? 87 : 178,
        r: 9,
        class: `${observed === 1 ? "class-one-mark" : "class-zero"}${incorrect ? " wrong-mark" : ""}`,
      });
    });
    addSvgElement(
      svg,
      "text",
      { x: left - 14, y: 91, "text-anchor": "end" },
      "real 1",
    );
    addSvgElement(
      svg,
      "text",
      { x: left - 14, y: 182, "text-anchor": "end" },
      "real 0",
    );
    addSvgElement(
      svg,
      "text",
      { x: right, y: 57, "text-anchor": "end" },
      "contorno vermelho = erro",
    );
  }

  const boundaryPoints = [
    [-2.1, -0.8, 0], [-1.8, 0.2, 0], [-1.5, 1.1, 0], [-1.2, -1.5, 0],
    [-0.8, 0.4, 0], [-0.3, -0.4, 0], [0.2, 0.5, 0], [0.8, -0.2, 0],
    [1.2, 0.8, 0], [1.5, -0.6, 0], [-2.4, 2.1, 1], [-1.1, 2.4, 1],
    [0.4, 2.2, 1], [1.8, 1.9, 1], [2.5, 0.8, 1], [2.4, -1.1, 1],
    [1.2, -2.2, 1], [-0.2, -2.4, 1], [-1.7, -2.1, 1], [-2.5, -1.5, 1],
  ];

  function drawBoundaryLab() {
    const model = byId("boundary-model").value;
    const quadraticStrength = Number(byId("boundary-curvature").value);
    byId("boundary-curvature").disabled = model === "linear";
    byId("boundary-curvature-label").textContent = formatNumber(
      quadraticStrength,
      1,
    );

    const svg = byId("boundary-chart");
    svg.replaceChildren();
    const left = 132;
    const right = 675;
    const top = 22;
    const bottom = 330;
    const scaleX = (value) => left + ((value + 3) / 6) * (right - left);
    const scaleY = (value) => bottom - ((value + 3) / 6) * (bottom - top);

    [-3, -2, -1, 0, 1, 2, 3].forEach((value) => {
      addLine(svg, scaleX(value), top, scaleX(value), bottom, "grid-line");
      addLine(svg, left, scaleY(value), right, scaleY(value), "grid-line");
    });
    addLine(svg, left, scaleY(0), right, scaleY(0), "axis");
    addLine(svg, scaleX(0), top, scaleX(0), bottom, "axis");
    addSvgElement(
      svg,
      "text",
      { x: right, y: scaleY(0) - 9, "text-anchor": "end" },
      "x₁",
    );
    addSvgElement(svg, "text", { x: scaleX(0) + 11, y: top + 13 }, "x₂");

    if (model === "linear") {
      addSvgElement(svg, "path", {
        d: `M${scaleX(-3)},${scaleY(-1.7)} L${scaleX(3)},${scaleY(1.7)}`,
        class: "main-curve",
      });
      byId("boundary-equation").textContent = "η = β₀ + β₁x₁ + β₂x₂";
      byId("boundary-interpretation").textContent =
        "Uma regressão logística com os atributos originais produz uma fronteira reta.";
    } else {
      const radius = 1.55 + 0.45 * quadraticStrength;
      addSvgElement(svg, "ellipse", {
        cx: scaleX(0),
        cy: scaleY(0),
        rx: ((right - left) / 6) * radius,
        ry: ((bottom - top) / 6) * radius,
        class: "main-curve",
      });
      byId("boundary-equation").textContent =
        "η = β₀ + β₁x₁ + β₂x₂ + γ(x₁² + x₂²)";
      byId("boundary-interpretation").textContent =
        "O modelo continua linear nos coeficientes; os atributos transformados tornam a fronteira curva no espaço original.";
    }

    boundaryPoints.forEach(([x, y, observed]) => {
      if (observed === 0) {
        addSvgElement(svg, "circle", {
          cx: scaleX(x),
          cy: scaleY(y),
          r: 8,
          class: "class-zero",
        });
      } else {
        addSvgElement(svg, "path", {
          d: `M${scaleX(x) - 8},${scaleY(y) - 8} L${scaleX(x) + 8},${scaleY(y) + 8} M${scaleX(x) + 8},${scaleY(y) - 8} L${scaleX(x) - 8},${scaleY(y) + 8}`,
          class: "class-one-mark",
        });
      }
    });

    addSvgElement(svg, "circle", {
      cx: 58,
      cy: 54,
      r: 7,
      class: "class-zero",
    });
    addSvgElement(svg, "text", { x: 75, y: 59 }, "classe 0");
    addSvgElement(svg, "path", {
      d: "M51,91 L65,105 M65,91 L51,105",
      class: "class-one-mark",
    });
    addSvgElement(svg, "text", { x: 75, y: 102 }, "classe 1");
  }

  ["logit-x", "logit-beta", "logit-beta0"].forEach((id) =>
    byId(id).addEventListener("input", drawLogitLab),
  );
  byId("threshold-value").addEventListener("input", drawThresholdLab);
  byId("boundary-model").addEventListener("change", drawBoundaryLab);
  byId("boundary-curvature").addEventListener("input", drawBoundaryLab);

  drawLogitLab();
  drawThresholdLab();
  drawBoundaryLab();
})();
