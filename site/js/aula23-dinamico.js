(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const fmt = (value, digits = 2) => Number(value).toLocaleString("pt-BR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

  function mulberry32(seed) {
    return function random() {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(random) {
    const u = Math.max(random(), 1e-12);
    const v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function solveLinearSystem(matrix, vector) {
    const n = vector.length;
    const a = matrix.map((row, i) => row.slice().concat(vector[i]));

    for (let col = 0; col < n; col += 1) {
      let pivot = col;
      for (let row = col + 1; row < n; row += 1) {
        if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
      }
      [a[col], a[pivot]] = [a[pivot], a[col]];
      if (Math.abs(a[col][col]) < 1e-12) a[col][col] = 1e-12;

      const divisor = a[col][col];
      for (let j = col; j <= n; j += 1) a[col][j] /= divisor;

      for (let row = 0; row < n; row += 1) {
        if (row === col) continue;
        const factor = a[row][col];
        for (let j = col; j <= n; j += 1) a[row][j] -= factor * a[col][j];
      }
    }
    return a.map((row) => row[n]);
  }

  function trueFunction(x) {
    return 0.65 + 1.15 * x - 1.25 * x * x + 0.55 * Math.sin(4.2 * x);
  }

  function generateRegressionSample(seed, n = 24) {
    const random = mulberry32(seed);
    const points = [];
    for (let i = 0; i < n; i += 1) {
      const base = -1 + (2 * i) / (n - 1);
      const x = clamp(base + (random() - 0.5) * 0.055, -1, 1);
      const y = trueFunction(x) + 0.27 * gaussian(random);
      points.push({ x, y });
    }
    return points.sort((a, b) => a.x - b.x);
  }

  function fitPolynomial(points, degree, lambda = 0) {
    const p = degree + 1;
    const xtx = Array.from({ length: p }, () => Array(p).fill(0));
    const xty = Array(p).fill(0);

    points.forEach(({ x, y }) => {
      const powers = Array(p).fill(1);
      for (let j = 1; j < p; j += 1) powers[j] = powers[j - 1] * x;
      for (let j = 0; j < p; j += 1) {
        xty[j] += powers[j] * y;
        for (let k = 0; k < p; k += 1) xtx[j][k] += powers[j] * powers[k];
      }
    });

    for (let j = 1; j < p; j += 1) xtx[j][j] += lambda * points.length;
    xtx[0][0] += 1e-10;
    return solveLinearSystem(xtx, xty);
  }

  function polynomialValue(coefficients, x) {
    let value = 0;
    for (let j = coefficients.length - 1; j >= 0; j -= 1) value = value * x + coefficients[j];
    return value;
  }

  function trainingMse(points, coefficients) {
    return points.reduce((sum, point) => {
      const error = point.y - polynomialValue(coefficients, point.x);
      return sum + error * error;
    }, 0) / points.length;
  }

  function expectedMse(coefficients) {
    let sum = 0;
    const count = 241;
    for (let i = 0; i < count; i += 1) {
      const x = -1 + (2 * i) / (count - 1);
      const error = trueFunction(x) - polynomialValue(coefficients, x);
      sum += error * error + 0.27 ** 2;
    }
    return sum / count;
  }

  function chartScales(width, height, xMin, xMax, yMin, yMax, margin) {
    return {
      x: (value) => margin.left + ((value - xMin) / (xMax - xMin)) * (width - margin.left - margin.right),
      y: (value) => height - margin.bottom - ((value - yMin) / (yMax - yMin)) * (height - margin.top - margin.bottom),
    };
  }

  function linePath(values, xScale, yScale) {
    return values.map((point, index) => `${index === 0 ? "M" : "L"}${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(2)}`).join(" ");
  }

  function regressionAxes(width, height, scales, margin) {
    const xTicks = [-1, -0.5, 0, 0.5, 1];
    const yTicks = [-2, -1, 0, 1, 2, 3];
    return `
      <g aria-hidden="true">
        ${yTicks.map((tick) => `<line class="grid-line" x1="${margin.left}" x2="${width - margin.right}" y1="${scales.y(tick)}" y2="${scales.y(tick)}"></line><text x="${margin.left - 12}" y="${scales.y(tick) + 4}" text-anchor="end">${fmt(tick, 0)}</text>`).join("")}
        ${xTicks.map((tick) => `<line class="grid-line" x1="${scales.x(tick)}" x2="${scales.x(tick)}" y1="${margin.top}" y2="${height - margin.bottom}"></line><text x="${scales.x(tick)}" y="${height - margin.bottom + 24}" text-anchor="middle">${fmt(tick, 1)}</text>`).join("")}
        <line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line>
        <line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
        <text x="${(margin.left + width - margin.right) / 2}" y="${height - 8}" text-anchor="middle">atributo x</text>
        <text x="18" y="${height / 2}" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">resposta y</text>
      </g>`;
  }

  function drawRegressionChart(svg, points, coefficients, clipId) {
    const width = 820;
    const height = 390;
    const margin = { top: 22, right: 22, bottom: 52, left: 62 };
    const scales = chartScales(width, height, -1.05, 1.05, -2.2, 3.2, margin);
    const grid = Array.from({ length: 260 }, (_, i) => {
      const x = -1.05 + (2.1 * i) / 259;
      return { x, trueY: trueFunction(x), fitY: polynomialValue(coefficients, x) };
    });
    const truePath = linePath(grid.map((d) => ({ x: d.x, y: d.trueY })), scales.x, scales.y);
    const fitPath = linePath(grid.map((d) => ({ x: d.x, y: clamp(d.fitY, -8, 8) })), scales.x, scales.y);

    svg.innerHTML = `
      <title>Ajuste polinomial e curva geradora</title>
      <desc>Os pontos mostram a amostra, a linha tracejada mostra a relação geradora e a linha laranja mostra o ajuste atual.</desc>
      <defs><clipPath id="${clipId}"><rect x="${margin.left}" y="${margin.top}" width="${width - margin.left - margin.right}" height="${height - margin.top - margin.bottom}"></rect></clipPath></defs>
      ${regressionAxes(width, height, scales, margin)}
      <g clip-path="url(#${clipId})">
        <path class="true-curve" d="${truePath}"></path>
        <path class="fit-curve" d="${fitPath}"></path>
        ${points.map((point) => `<circle class="sample-point" cx="${scales.x(point.x)}" cy="${scales.y(point.y)}" r="5"><title>x=${fmt(point.x)}, y=${fmt(point.y)}</title></circle>`).join("")}
      </g>
      <g transform="translate(${width - 236},${margin.top + 2})" aria-hidden="true">
        <line class="true-curve" x1="0" x2="32" y1="0" y2="0"></line><text x="42" y="5">relação geradora</text>
        <line class="fit-curve" x1="0" x2="32" y1="24" y2="24"></line><text x="42" y="29">modelo ajustado</text>
      </g>`;
  }

  function initializeComplexityLab() {
    const svg = $("complexity-chart");
    if (!svg) return;
    let seed = 2301;
    let sample = generateRegressionSample(seed);
    const degreeInput = $("complexity-degree");

    function update() {
      const degree = Number(degreeInput.value);
      const coefficients = fitPolynomial(sample, degree, 1e-10);
      const train = trainingMse(sample, coefficients);
      const valid = expectedMse(coefficients);
      const risks = [];
      for (let d = 1; d <= 12; d += 1) risks.push(expectedMse(fitPolynomial(sample, d, 1e-10)));
      const bestDegree = risks.indexOf(Math.min(...risks)) + 1;
      let status = "equilíbrio razoável";
      if (degree < bestDegree - 1) status = "subajuste provável";
      if (degree > bestDegree + 2) status = "sobreajuste provável";

      $("complexity-degree-value").textContent = degree;
      $("complexity-train-mse").textContent = fmt(train, 3);
      $("complexity-valid-mse").textContent = fmt(valid, 3);
      $("complexity-status").textContent = status;
      $("complexity-interpretation").textContent = degree > bestDegree + 2
        ? `O treino continua melhorando, mas a oscilação fora dos pontos aumenta. Nesta amostra, a validação favorece aproximadamente o grau ${bestDegree}.`
        : `Nesta amostra, o menor erro de validação ocorre aproximadamente no grau ${bestDegree}. O grau escolhido não deve ser avaliado apenas pelo ajuste ao treino.`;
      drawRegressionChart(svg, sample, coefficients, "complexity-clip");
    }

    degreeInput.addEventListener("input", update);
    $("complexity-resample").addEventListener("click", () => {
      seed += 97;
      sample = generateRegressionSample(seed);
      update();
    });
    update();
  }

  function initializeRidgeLab() {
    const svg = $("ridge-chart");
    if (!svg) return;
    const sample = generateRegressionSample(9182);
    const lambdaValues = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10];
    const lambdaInput = $("ridge-lambda");

    function lambdaLabel(lambda) {
      if (lambda === 0) return "0";
      if (lambda >= 0.01) return fmt(lambda, lambda < 1 ? 2 : 0);
      return lambda.toExponential(0).replace("e-", " × 10⁻").replace(/0(\d)/, "$1");
    }

    function update() {
      const lambda = lambdaValues[Number(lambdaInput.value)];
      const coefficients = fitPolynomial(sample, 12, lambda);
      const norm = Math.sqrt(coefficients.slice(1).reduce((sum, value) => sum + value * value, 0));
      const train = trainingMse(sample, coefficients);
      const valid = expectedMse(coefficients);

      $("ridge-lambda-value").textContent = lambdaLabel(lambda);
      $("ridge-coef-norm").textContent = fmt(norm, 2);
      $("ridge-train-mse").textContent = fmt(train, 3);
      $("ridge-valid-mse").textContent = fmt(valid, 3);
      if (lambda === 0) {
        $("ridge-interpretation").textContent = "Sem penalização, o polinômio de grau 12 tenta acompanhar detalhes particulares desta amostra.";
      } else if (lambda < 0.01) {
        $("ridge-interpretation").textContent = "Uma penalização pequena já reduz coeficientes extremos e suaviza as oscilações mais instáveis.";
      } else if (lambda <= 1) {
        $("ridge-interpretation").textContent = "A penalização produz um compromisso entre acompanhar a estrutura e evitar coeficientes excessivos.";
      } else {
        $("ridge-interpretation").textContent = "Com λ muito grande, o viés passa a dominar: a curva fica estável, mas simples demais.";
      }
      drawRegressionChart(svg, sample, coefficients, "ridge-clip");
    }

    lambdaInput.addEventListener("input", update);
    update();
  }

  function geometryObjective(x, y) {
    const dx = x - 2.3;
    const dy = y - 1.35;
    return dx * dx + 1.64 * dx * dy + dy * dy;
  }

  function geometryBoundary(type, radius, count = 720) {
    const points = [];
    if (type === "l2") {
      for (let i = 0; i < count; i += 1) {
        const angle = (2 * Math.PI * i) / count;
        points.push({ x: radius * Math.cos(angle), y: radius * Math.sin(angle) });
      }
    } else {
      for (let i = 0; i < count / 2; i += 1) {
        const x = -radius + (2 * radius * i) / (count / 2 - 1);
        const y = radius - Math.abs(x);
        points.push({ x, y }, { x, y: -y });
      }
    }
    return points;
  }

  function geometryContour(level) {
    const points = [];
    const root2 = Math.sqrt(2);
    const lambdaAlong = 1.82;
    const lambdaAcross = 0.18;
    for (let i = 0; i <= 240; i += 1) {
      const angle = (2 * Math.PI * i) / 240;
      const along = Math.sqrt(level / lambdaAlong) * Math.cos(angle);
      const across = Math.sqrt(level / lambdaAcross) * Math.sin(angle);
      points.push({
        x: 2.3 + (along + across) / root2,
        y: 1.35 + (along - across) / root2,
      });
    }
    return points;
  }

  function initializeGeometryLab() {
    const svg = $("geometry-chart");
    if (!svg) return;
    const typeInput = $("geometry-penalty");
    const radiusInput = $("geometry-radius");

    function update() {
      const type = typeInput.value;
      const radius = Number(radiusInput.value);
      const boundary = geometryBoundary(type, radius);
      const solution = boundary.reduce((best, point) => geometryObjective(point.x, point.y) < geometryObjective(best.x, best.y) ? point : best, boundary[0]);
      const contactLevel = geometryObjective(solution.x, solution.y);
      const width = 720;
      const height = 520;
      const margin = { top: 24, right: 28, bottom: 58, left: 70 };
      const scales = chartScales(width, height, -3.4, 3.4, -3.4, 3.4, margin);
      const levels = [0.25, 0.75, 1.5, 3, 5, 8, contactLevel]
        .filter((value) => value > 0.03)
        .sort((a, b) => a - b);
      const regionPath = type === "l1"
        ? `M${scales.x(radius)},${scales.y(0)} L${scales.x(0)},${scales.y(radius)} L${scales.x(-radius)},${scales.y(0)} L${scales.x(0)},${scales.y(-radius)} Z`
        : `<ellipse class="constraint-region" cx="${scales.x(0)}" cy="${scales.y(0)}" rx="${Math.abs(scales.x(radius) - scales.x(0))}" ry="${Math.abs(scales.y(radius) - scales.y(0))}"></ellipse>`;
      const ticks = [-3, -2, -1, 0, 1, 2, 3];

      svg.innerHTML = `
        <title>Geometria das restrições ${type.toUpperCase()}</title>
        <desc>Curvas elípticas representam níveis da perda; a região azul é o conjunto permitido e o ponto laranja é a solução regularizada.</desc>
        <g aria-hidden="true">
          ${ticks.map((tick) => `<line class="grid-line" x1="${scales.x(tick)}" x2="${scales.x(tick)}" y1="${margin.top}" y2="${height - margin.bottom}"></line><line class="grid-line" x1="${margin.left}" x2="${width - margin.right}" y1="${scales.y(tick)}" y2="${scales.y(tick)}"></line>`).join("")}
          ${ticks.map((tick) => `<text x="${scales.x(tick)}" y="${height - margin.bottom + 24}" text-anchor="middle">${tick}</text><text x="${margin.left - 12}" y="${scales.y(tick) + 4}" text-anchor="end">${tick}</text>`).join("")}
          <line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${scales.y(0)}" y2="${scales.y(0)}"></line>
          <line class="axis" x1="${scales.x(0)}" x2="${scales.x(0)}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
          <text x="${(margin.left + width - margin.right) / 2}" y="${height - 10}" text-anchor="middle">coeficiente β₁</text>
          <text x="18" y="${height / 2}" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">coeficiente β₂</text>
        </g>
        ${levels.map((level) => `<path class="loss-contour" d="${linePath(geometryContour(level), scales.x, scales.y)}"></path>`).join("")}
        ${type === "l1" ? `<path class="constraint-region" d="${regionPath}"></path>` : regionPath}
        <circle class="unconstrained-point" cx="${scales.x(2.3)}" cy="${scales.y(1.35)}" r="6"><title>Solução sem restrição</title></circle>
        <text x="${scales.x(2.3) + 10}" y="${scales.y(1.35) - 10}">mínimo sem penalização</text>
        <circle class="solution-point" cx="${scales.x(solution.x)}" cy="${scales.y(solution.y)}" r="8"><title>Solução regularizada: beta 1 = ${fmt(solution.x)}, beta 2 = ${fmt(solution.y)}</title></circle>`;

      const zeroed = Math.abs(solution.x) < 0.035 || Math.abs(solution.y) < 0.035;
      $("geometry-radius-value").textContent = fmt(radius, 1);
      $("geometry-beta1").textContent = fmt(solution.x, 2);
      $("geometry-beta2").textContent = fmt(solution.y, 2);
      $("geometry-status").textContent = zeroed ? "um coeficiente zerado" : "dois coeficientes ativos";
      $("geometry-interpretation").textContent = type === "l1"
        ? (zeroed ? "A menor elipse toca um vértice da região L1. Nesse ponto, um coeficiente é exatamente zero." : "Com este raio, a solução L1 ainda está em uma aresta. Reduza o raio e observe a aproximação de um vértice.")
        : "A borda suave da região L2 contrai os dois coeficientes, mas dificilmente encontra um eixo exatamente.";
    }

    typeInput.addEventListener("change", update);
    radiusInput.addEventListener("input", update);
    update();
  }

  function centerAndScale(data) {
    const meanX1 = data.reduce((sum, row) => sum + row.x1, 0) / data.length;
    const meanX2 = data.reduce((sum, row) => sum + row.x2, 0) / data.length;
    const meanY = data.reduce((sum, row) => sum + row.y, 0) / data.length;
    const sdX1 = Math.sqrt(data.reduce((sum, row) => sum + (row.x1 - meanX1) ** 2, 0) / data.length) || 1;
    const sdX2 = Math.sqrt(data.reduce((sum, row) => sum + (row.x2 - meanX2) ** 2, 0) / data.length) || 1;
    return data.map((row) => ({
      x1: (row.x1 - meanX1) / sdX1,
      x2: (row.x2 - meanX2) / sdX2,
      y: row.y - meanY,
    }));
  }

  function fitTwoPredictors(data, method) {
    const rows = centerAndScale(data);
    const n = rows.length;
    if (method === "lasso") {
      let b1 = 0;
      let b2 = 0;
      const lambda = 0.7;
      const soft = (value, threshold) => Math.sign(value) * Math.max(Math.abs(value) - threshold, 0);
      for (let iteration = 0; iteration < 120; iteration += 1) {
        const raw1 = rows.reduce((sum, row) => sum + row.x1 * (row.y - b2 * row.x2), 0) / n;
        b1 = soft(raw1, lambda);
        const raw2 = rows.reduce((sum, row) => sum + row.x2 * (row.y - b1 * row.x1), 0) / n;
        b2 = soft(raw2, lambda);
      }
      return [b1, b2];
    }

    const xtx = [[0, 0], [0, 0]];
    const xty = [0, 0];
    rows.forEach((row) => {
      xtx[0][0] += row.x1 * row.x1;
      xtx[0][1] += row.x1 * row.x2;
      xtx[1][0] += row.x1 * row.x2;
      xtx[1][1] += row.x2 * row.x2;
      xty[0] += row.x1 * row.y;
      xty[1] += row.x2 * row.y;
    });
    const ridgeLambda = method === "ridge" ? 0.22 * n : 1e-8;
    xtx[0][0] += ridgeLambda;
    xtx[1][1] += ridgeLambda;
    return solveLinearSystem(xtx, xty);
  }

  function generateCollinearData(seed, rho, n = 90) {
    const random = mulberry32(seed);
    const data = [];
    for (let i = 0; i < n; i += 1) {
      const x1 = gaussian(random);
      const x2 = rho * x1 + Math.sqrt(Math.max(1 - rho * rho, 0.0001)) * gaussian(random);
      const y = 5 * x1 + 1.25 * gaussian(random);
      data.push({ x1, x2, y });
    }
    return data;
  }

  function bootstrapCoefficients(data, method, seed, count = 36) {
    const random = mulberry32(seed);
    const estimates = [];
    for (let b = 0; b < count; b += 1) {
      const sample = Array.from({ length: data.length }, () => data[Math.floor(random() * data.length)]);
      const [b1, b2] = fitTwoPredictors(sample, method);
      estimates.push({ b1, b2, method });
    }
    return estimates;
  }

  function coefficientInstability(estimates) {
    const mean1 = estimates.reduce((sum, row) => sum + row.b1, 0) / estimates.length;
    const mean2 = estimates.reduce((sum, row) => sum + row.b2, 0) / estimates.length;
    const variance1 = estimates.reduce((sum, row) => sum + (row.b1 - mean1) ** 2, 0) / estimates.length;
    const variance2 = estimates.reduce((sum, row) => sum + (row.b2 - mean2) ** 2, 0) / estimates.length;
    return Math.sqrt(variance1 + variance2);
  }

  function initializeCollinearityLab() {
    const svg = $("collinearity-chart");
    if (!svg) return;
    const rhoInput = $("collinearity-rho");
    let seed = 6217;

    function update() {
      const rho = Number(rhoInput.value);
      const data = generateCollinearData(seed, rho);
      const ols = bootstrapCoefficients(data, "ols", seed + 11);
      const ridge = bootstrapCoefficients(data, "ridge", seed + 29);
      const lasso = bootstrapCoefficients(data, "lasso", seed + 47);
      const all = [...ols, ...ridge, ...lasso];
      const values = all.flatMap((row) => [row.b1, row.b2]).concat([0, 5]);
      const rawMin = Math.min(...values);
      const rawMax = Math.max(...values);
      const domainMin = Math.max(-12, Math.floor(rawMin - 0.8));
      const domainMax = Math.min(18, Math.ceil(rawMax + 0.8));
      const width = 720;
      const height = 520;
      const margin = { top: 28, right: 28, bottom: 62, left: 72 };
      const scales = chartScales(width, height, domainMin, domainMax, domainMin, domainMax, margin);
      const ticks = Array.from({ length: 6 }, (_, i) => domainMin + ((domainMax - domainMin) * i) / 5);

      const pointMarkup = all.map((row) => {
        const x = scales.x(row.b1);
        const y = scales.y(row.b2);
        const title = `${row.method}: β₁=${fmt(row.b1)}, β₂=${fmt(row.b2)}`;
        if (row.method === "ridge") return `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" fill="var(--reg-blue)" opacity="0.72"><title>${title}</title></rect>`;
        if (row.method === "lasso") return `<path d="M${x},${y - 5} L${x + 5},${y + 5} L${x - 5},${y + 5} Z" fill="var(--reg-purple)" opacity="0.72"><title>${title}</title></path>`;
        return `<circle cx="${x}" cy="${y}" r="4" fill="var(--reg-red)" opacity="0.62"><title>${title}</title></circle>`;
      }).join("");

      svg.innerHTML = `
        <title>Estabilidade dos coeficientes sob colinearidade</title>
        <desc>Cada símbolo é um par de coeficientes estimado em uma amostra bootstrap. Nuvens mais compactas indicam maior estabilidade.</desc>
        <g aria-hidden="true">
          ${ticks.map((tick) => `<line class="grid-line" x1="${scales.x(tick)}" x2="${scales.x(tick)}" y1="${margin.top}" y2="${height - margin.bottom}"></line><line class="grid-line" x1="${margin.left}" x2="${width - margin.right}" y1="${scales.y(tick)}" y2="${scales.y(tick)}"></line>`).join("")}
          ${ticks.map((tick) => `<text x="${scales.x(tick)}" y="${height - margin.bottom + 24}" text-anchor="middle">${fmt(tick, 1)}</text><text x="${margin.left - 12}" y="${scales.y(tick) + 4}" text-anchor="end">${fmt(tick, 1)}</text>`).join("")}
          <line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line>
          <line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
          <line x1="${scales.x(domainMin)}" x2="${scales.x(domainMax)}" y1="${scales.y(5 - domainMin)}" y2="${scales.y(5 - domainMax)}" stroke="#9aa9b1" stroke-dasharray="7 5" stroke-width="2"></line>
          <text x="${(margin.left + width - margin.right) / 2}" y="${height - 10}" text-anchor="middle">coeficiente β₁</text>
          <text x="18" y="${height / 2}" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">coeficiente β₂</text>
        </g>
        ${pointMarkup}
        <circle cx="${scales.x(5)}" cy="${scales.y(0)}" r="7" fill="#455761" stroke="#fff" stroke-width="2"><title>Coeficientes geradores: 5 e 0</title></circle>
        <text x="${scales.x(5) + 10}" y="${scales.y(0) - 10}">referência (5, 0)</text>`;

      const lassoZeros = lasso.reduce((sum, row) => sum + Number(Math.abs(row.b1) < 0.04) + Number(Math.abs(row.b2) < 0.04), 0);
      $("collinearity-rho-value").textContent = fmt(rho, 2);
      $("collinearity-ols-instability").textContent = fmt(coefficientInstability(ols), 2);
      $("collinearity-ridge-instability").textContent = fmt(coefficientInstability(ridge), 2);
      $("collinearity-lasso-zeros").textContent = `${lassoZeros} de ${lasso.length * 2}`;
      $("collinearity-interpretation").textContent = rho > 0.85
        ? "Com forte colinearidade, OLS espalha o mesmo efeito total por combinações muito diferentes. Ridge concentra a nuvem; Lasso pode concentrar o efeito em apenas um dos preditores equivalentes."
        : "Com menor correlação, os dados distinguem melhor os efeitos dos dois preditores e as estimativas OLS ficam menos dispersas.";
    }

    rhoInput.addEventListener("input", update);
    $("collinearity-resample").addEventListener("click", () => {
      seed += 131;
      update();
    });
    update();
  }

  initializeComplexityLab();
  initializeRidgeLab();
  initializeGeometryLab();
  initializeCollinearityLab();
})();
