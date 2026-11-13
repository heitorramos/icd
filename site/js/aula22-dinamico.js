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

  const integratedState = {
    patients: [],
    threshold: 0.5,
    hoveredPatient: null,
    seed: 22026,
  };
  const integratedBeta0 = -4.2;
  const integratedBeta1 = 0.85;

  function integratedRandom() {
    integratedState.seed =
      (1664525 * integratedState.seed + 1013904223) >>> 0;
    return integratedState.seed / 4294967296;
  }

  function generateIntegratedCohort() {
    integratedState.patients = Array.from({ length: 60 }, (_, index) => {
      const biomarker = 0.5 + 9 * integratedRandom();
      const probability = sigmoid(
        integratedBeta0 + integratedBeta1 * biomarker,
      );
      return {
        id: index + 1,
        biomarker,
        probability,
        observed: integratedRandom() < probability ? 1 : 0,
        jitter: (integratedRandom() - 0.5) * 0.045,
        pixelX: 0,
        pixelY: 0,
      };
    });
    drawIntegratedLab();
  }

  function evaluateIntegratedThreshold(threshold) {
    let truePositive = 0;
    let trueNegative = 0;
    let falsePositive = 0;
    let falseNegative = 0;
    integratedState.patients.forEach(({ probability, observed }) => {
      const predicted = probability >= threshold ? 1 : 0;
      if (observed === 1 && predicted === 1) truePositive += 1;
      if (observed === 0 && predicted === 0) trueNegative += 1;
      if (observed === 0 && predicted === 1) falsePositive += 1;
      if (observed === 1 && predicted === 0) falseNegative += 1;
    });
    const precisionDenominator = truePositive + falsePositive;
    const recallDenominator = truePositive + falseNegative;
    const precision = precisionDenominator
      ? truePositive / precisionDenominator
      : 0;
    const recall = recallDenominator ? truePositive / recallDenominator : 0;
    const accuracy =
      (truePositive + trueNegative) /
      Math.max(1, integratedState.patients.length);
    const f1 = precision + recall
      ? (2 * precision * recall) / (precision + recall)
      : 0;
    return {
      truePositive,
      trueNegative,
      falsePositive,
      falseNegative,
      precision,
      recall,
      accuracy,
      f1,
    };
  }

  function prepareCanvas(canvas) {
    const bounds = canvas.parentElement.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(bounds.width * ratio));
    canvas.height = Math.max(1, Math.round(bounds.height * ratio));
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width: bounds.width, height: bounds.height };
  }

  function drawIntegratedSigmoid() {
    const canvas = byId("integrated-sigmoid-chart");
    const { context, width, height } = prepareCanvas(canvas);
    const padding = { top: 28, right: 24, bottom: 50, left: 58 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const scaleX = (value) => padding.left + (value / 10) * plotWidth;
    const scaleY = (value) => padding.top + (1 - value) * plotHeight;
    const thresholdY = scaleY(integratedState.threshold);

    context.clearRect(0, 0, width, height);
    context.fillStyle = "rgba(217, 95, 2, 0.06)";
    context.fillRect(
      padding.left,
      padding.top,
      plotWidth,
      Math.max(0, thresholdY - padding.top),
    );
    context.fillStyle = "rgba(15, 107, 120, 0.05)";
    context.fillRect(
      padding.left,
      thresholdY,
      plotWidth,
      Math.max(0, padding.top + plotHeight - thresholdY),
    );

    context.strokeStyle = "#d9e2e6";
    context.lineWidth = 1;
    context.fillStyle = "#5d6b75";
    context.font = '12px "Atkinson Hyperlegible", Arial, sans-serif';
    [0, 0.25, 0.5, 0.75, 1].forEach((value) => {
      const y = scaleY(value);
      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(padding.left + plotWidth, y);
      context.stroke();
      context.textAlign = "right";
      context.fillText(formatNumber(value, 2), padding.left - 8, y + 4);
    });
    [0, 2, 4, 6, 8, 10].forEach((value) => {
      const x = scaleX(value);
      context.beginPath();
      context.moveTo(x, padding.top);
      context.lineTo(x, padding.top + plotHeight);
      context.stroke();
      context.textAlign = "center";
      context.fillText(String(value), x, padding.top + plotHeight + 19);
    });

    context.fillStyle = "#455761";
    context.font = '600 12px "Atkinson Hyperlegible", Arial, sans-serif';
    context.textAlign = "center";
    context.fillText(
      "biomarcador sintético x",
      padding.left + plotWidth / 2,
      height - 8,
    );
    context.save();
    context.translate(16, padding.top + plotHeight / 2);
    context.rotate(-Math.PI / 2);
    context.fillText("probabilidade estimada", 0, 0);
    context.restore();

    context.beginPath();
    context.strokeStyle = "#17324d";
    context.lineWidth = 3;
    for (let index = 0; index <= 200; index += 1) {
      const biomarker = (10 * index) / 200;
      const probability = sigmoid(
        integratedBeta0 + integratedBeta1 * biomarker,
      );
      const x = scaleX(biomarker);
      const y = scaleY(probability);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();

    context.save();
    context.beginPath();
    context.setLineDash([7, 5]);
    context.strokeStyle = "#d95f02";
    context.lineWidth = 2.5;
    context.moveTo(padding.left, thresholdY);
    context.lineTo(padding.left + plotWidth, thresholdY);
    context.stroke();
    context.restore();
    context.fillStyle = "#d95f02";
    context.font = '700 12px "Atkinson Hyperlegible", Arial, sans-serif';
    context.textAlign = "left";
    context.fillText(
      `c = ${formatNumber(integratedState.threshold)}`,
      padding.left + 8,
      Math.max(padding.top + 14, thresholdY - 8),
    );

    integratedState.patients.forEach((patient) => {
      const x = scaleX(patient.biomarker);
      const visibleProbability = Math.min(
        0.985,
        Math.max(0.015, patient.probability + patient.jitter),
      );
      const y = scaleY(visibleProbability);
      const predicted =
        patient.probability >= integratedState.threshold ? 1 : 0;
      const incorrect = predicted !== patient.observed;
      const hovered = integratedState.hoveredPatient?.id === patient.id;
      patient.pixelX = x;
      patient.pixelY = y;
      context.beginPath();
      context.arc(x, y, hovered ? 8.5 : 6, 0, Math.PI * 2);
      context.fillStyle = patient.observed ? "#d95f02" : "#0f6b78";
      context.fill();
      context.lineWidth = incorrect ? 3 : 1.5;
      context.strokeStyle = incorrect ? "#b42318" : "#ffffff";
      context.stroke();
    });
  }

  function drawIntegratedTradeoff() {
    const canvas = byId("integrated-tradeoff-chart");
    const { context, width, height } = prepareCanvas(canvas);
    const padding = { top: 14, right: 14, bottom: 28, left: 38 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const scaleX = (value) =>
      padding.left + ((value - 0.05) / 0.9) * plotWidth;
    const scaleY = (value) => padding.top + (1 - value) * plotHeight;
    context.clearRect(0, 0, width, height);
    context.font = '11px "Atkinson Hyperlegible", Arial, sans-serif';
    context.strokeStyle = "#d9e2e6";
    context.fillStyle = "#5d6b75";
    [0, 0.5, 1].forEach((value) => {
      const y = scaleY(value);
      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(padding.left + plotWidth, y);
      context.stroke();
      context.textAlign = "right";
      context.fillText(`${Math.round(value * 100)}%`, padding.left - 5, y + 3);
    });
    [0.1, 0.5, 0.9].forEach((value) => {
      context.textAlign = "center";
      context.fillText(
        formatNumber(value, 1),
        scaleX(value),
        height - 6,
      );
    });

    const points = Array.from({ length: 46 }, (_, index) => {
      const threshold = 0.05 + (0.9 * index) / 45;
      return { threshold, ...evaluateIntegratedThreshold(threshold) };
    });
    [
      { key: "precision", color: "#0f6b78" },
      { key: "recall", color: "#237a57" },
    ].forEach(({ key, color }) => {
      context.beginPath();
      context.strokeStyle = color;
      context.lineWidth = 2.5;
      points.forEach((point, index) => {
        const x = scaleX(point.threshold);
        const y = scaleY(point[key]);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
    });

    const currentX = scaleX(integratedState.threshold);
    context.save();
    context.beginPath();
    context.setLineDash([5, 4]);
    context.strokeStyle = "#d95f02";
    context.lineWidth = 2;
    context.moveTo(currentX, padding.top);
    context.lineTo(currentX, padding.top + plotHeight);
    context.stroke();
    context.restore();
    context.fillStyle = "#0f6b78";
    context.textAlign = "left";
    context.fillText("precisão", padding.left + 6, padding.top + 12);
    context.fillStyle = "#237a57";
    context.fillText("revocação", padding.left + 6, padding.top + 26);
  }

  function updateIntegratedMetrics() {
    const result = evaluateIntegratedThreshold(integratedState.threshold);
    const percentage = (value) => `${formatNumber(100 * value, 1)}%`;
    byId("integrated-threshold-label").textContent = formatNumber(
      integratedState.threshold,
    );
    byId("integrated-tp").textContent = result.truePositive;
    byId("integrated-tn").textContent = result.trueNegative;
    byId("integrated-fp").textContent = result.falsePositive;
    byId("integrated-fn").textContent = result.falseNegative;
    byId("integrated-accuracy").textContent =
      `Acurácia: ${percentage(result.accuracy)}`;
    byId("integrated-precision").textContent = percentage(result.precision);
    byId("integrated-recall").textContent = percentage(result.recall);
    byId("integrated-f1").textContent = percentage(result.f1);
    byId("integrated-precision-bar").style.width =
      `${100 * result.precision}%`;
    byId("integrated-recall-bar").style.width = `${100 * result.recall}%`;

    let heading;
    let explanation;
    if (integratedState.threshold <= 0.3) {
      heading = "Limiar baixo: prioridade para detectar casos positivos.";
      explanation = `A revocação chegou a ${percentage(result.recall)}, mas ${result.falsePositive} pacientes da classe 0 foram classificados como positivos.`;
    } else if (integratedState.threshold >= 0.7) {
      heading = "Limiar alto: exige-se mais evidência para prever a classe 1.";
      explanation = `A precisão chegou a ${percentage(result.precision)}, enquanto ${result.falseNegative} pacientes da classe 1 deixaram de ser detectados.`;
    } else {
      heading = "Limiar intermediário: os dois tipos de erro coexistem.";
      explanation = `Há ${result.falsePositive} falsos positivos e ${result.falseNegative} falsos negativos. O F1-score é ${percentage(result.f1)}.`;
    }
    byId("integrated-feedback").innerHTML =
      `<strong>${heading}</strong> ${explanation} O melhor compromisso depende do custo de cada erro no problema real.`;
  }

  function drawIntegratedLab() {
    updateIntegratedMetrics();
    drawIntegratedSigmoid();
    drawIntegratedTradeoff();
  }

  function inspectIntegratedPoint(clientX, clientY) {
    const canvas = byId("integrated-sigmoid-chart");
    const bounds = canvas.getBoundingClientRect();
    const pointerX = clientX - bounds.left;
    const pointerY = clientY - bounds.top;
    let nearest = null;
    let minimumDistance = 16;
    integratedState.patients.forEach((patient) => {
      const distance = Math.hypot(
        patient.pixelX - pointerX,
        patient.pixelY - pointerY,
      );
      if (distance < minimumDistance) {
        minimumDistance = distance;
        nearest = patient;
      }
    });
    integratedState.hoveredPatient = nearest;
    const tooltip = byId("integrated-tooltip");
    if (!nearest) {
      tooltip.hidden = true;
      drawIntegratedSigmoid();
      return;
    }
    const predicted =
      nearest.probability >= integratedState.threshold ? 1 : 0;
    const status =
      nearest.observed === 1 && predicted === 1
        ? "verdadeiro positivo"
        : nearest.observed === 1
          ? "falso negativo"
          : predicted === 1
            ? "falso positivo"
            : "verdadeiro negativo";
    tooltip.innerHTML =
      `<strong>Paciente ${nearest.id}</strong><br>` +
      `x = ${formatNumber(nearest.biomarker, 2)}<br>` +
      `π̂ = ${formatNumber(nearest.probability, 2)}<br>` +
      `real = ${nearest.observed}; previsto = ${predicted}<br>${status}`;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(bounds.width - 220, Math.max(8, pointerX + 12))}px`;
    tooltip.style.top = `${Math.min(bounds.height - 118, Math.max(8, pointerY - 10))}px`;
    drawIntegratedSigmoid();
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

  byId("integrated-threshold").addEventListener("input", (event) => {
    integratedState.threshold = Number(event.target.value);
    drawIntegratedLab();
  });
  document.querySelectorAll(".integrated-preset").forEach((button) => {
    button.addEventListener("click", () => {
      integratedState.threshold = Number(button.dataset.threshold);
      byId("integrated-threshold").value = integratedState.threshold;
      drawIntegratedLab();
    });
  });
  byId("integrated-new-cohort").addEventListener(
    "click",
    generateIntegratedCohort,
  );
  byId("integrated-sigmoid-chart").addEventListener(
    "pointermove",
    (event) => inspectIntegratedPoint(event.clientX, event.clientY),
  );
  byId("integrated-sigmoid-chart").addEventListener(
    "pointerdown",
    (event) => inspectIntegratedPoint(event.clientX, event.clientY),
  );
  byId("integrated-sigmoid-chart").addEventListener("pointerleave", () => {
    integratedState.hoveredPatient = null;
    byId("integrated-tooltip").hidden = true;
    drawIntegratedSigmoid();
  });
  ["logit-x", "logit-beta", "logit-beta0"].forEach((id) =>
    byId(id).addEventListener("input", drawLogitLab),
  );
  byId("threshold-value").addEventListener("input", drawThresholdLab);
  byId("boundary-model").addEventListener("change", drawBoundaryLab);
  byId("boundary-curvature").addEventListener("input", drawBoundaryLab);
  window.addEventListener("resize", () => {
    drawIntegratedSigmoid();
    drawIntegratedTradeoff();
  });

  generateIntegratedCohort();
  drawLogitLab();
  drawThresholdLab();
  drawBoundaryLab();
})();
