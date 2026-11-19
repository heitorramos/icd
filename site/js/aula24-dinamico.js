(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const fmt = (value, digits = 3) => Number(value).toLocaleString("pt-BR", {
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

  function initializeFeatureLab() {
    if (!$("feature-lab")) return;
    const labels = {
      entire: "Apartamento inteiro",
      private: "Quarto privativo",
      shared: "Quarto compartilhado",
      bronx: "Bronx",
      brooklyn: "Brooklyn",
      manhattan: "Manhattan",
      queens: "Queens",
      staten: "Staten Island",
    };

    function update() {
      const room = $("feature-room").value;
      const region = $("feature-region").value;
      const nights = Number($("feature-nights").value);
      const reviewsRaw = $("feature-reviews").value;
      const missing = reviewsRaw === "missing";
      const reviews = missing ? 0.72 : Number(reviewsRaw);
      const useLog = $("feature-log").checked;
      const useIndicator = $("feature-missing-indicator").checked;

      $("feature-nights-value").textContent = `${nights} ${nights === 1 ? "noite" : "noites"}`;
      $("feature-raw-values").innerHTML = `
        <dt>acomodação</dt><dd>${labels[room]}</dd>
        <dt>região</dt><dd>${labels[region]}</dd>
        <dt>estadia mínima</dt><dd>${nights}</dd>
        <dt>avaliações/mês</dt><dd>${missing ? "ausente" : fmt(reviews, 1)}</dd>`;

      const features = [
        ["intercepto", 1],
        [useLog ? "log1p(noites)" : "noites", useLog ? Math.log1p(nights) : nights],
        ["avaliações imputadas", reviews],
      ];
      if (useIndicator) features.push(["avaliações ausentes", missing ? 1 : 0]);
      features.push(
        ["acomodação inteira", room === "entire" ? 1 : 0],
        ["acomodação privativa", room === "private" ? 1 : 0],
        ["região Bronx", region === "bronx" ? 1 : 0],
        ["região Brooklyn", region === "brooklyn" ? 1 : 0],
        ["região Manhattan", region === "manhattan" ? 1 : 0],
        ["região Queens", region === "queens" ? 1 : 0],
      );

      $("feature-vector").innerHTML = features.map(([name, value]) => `
        <div><span>${name}</span><strong>${Number.isInteger(value) ? value : fmt(value, 2)}</strong></div>`).join("");
      $("feature-dimension").textContent = `p = ${features.length}`;

      const pieces = [];
      pieces.push(useLog ? "A transformação log comprime noites muito grandes." : "Sem log, noites grandes permanecem dominantes na escala.");
      if (missing && useIndicator) pieces.push("A imputação fornece 0,72 e o indicador preserva a informação de que o valor estava ausente.");
      if (missing && !useIndicator) pieces.push("A imputação fornece 0,72, mas o modelo não distingue esse valor de uma observação realmente igual à mediana.");
      if (!missing) pieces.push("Como o valor foi observado, o indicador de ausência vale zero quando está presente.");
      $("feature-interpretation").textContent = pieces.join(" ");
    }

    ["feature-room", "feature-region", "feature-reviews", "feature-nights", "feature-log", "feature-missing-indicator"]
      .forEach((id) => $(id).addEventListener("input", update));
    update();
  }

  function targetEncodingMse(seed, perNeighborhood) {
    const random = mulberry32(seed);
    const effects = [-42, -30, -18, -8, 2, 12, 24, 38, 52, 68];
    const rows = [];
    effects.forEach((effect, neighborhood) => {
      for (let j = 0; j < perNeighborhood; j += 1) {
        rows.push({
          neighborhood,
          y: 150 + effect + 22 * gaussian(random),
          fold: (neighborhood * 2 + j * 3) % 5,
        });
      }
    });

    let correctSquared = 0;
    let leakySquared = 0;
    rows.forEach((row) => {
      const train = rows.filter((other) => other.fold !== row.fold);
      const sameTrain = train.filter((other) => other.neighborhood === row.neighborhood);
      const sameAll = rows.filter((other) => other.neighborhood === row.neighborhood);
      const globalTrain = train.reduce((sum, item) => sum + item.y, 0) / train.length;
      const correctMean = sameTrain.length
        ? sameTrain.reduce((sum, item) => sum + item.y, 0) / sameTrain.length
        : globalTrain;
      const leakyMean = sameAll.reduce((sum, item) => sum + item.y, 0) / sameAll.length;
      correctSquared += (row.y - correctMean) ** 2;
      leakySquared += (row.y - leakyMean) ** 2;
    });
    return {
      correct: correctSquared / rows.length,
      leaky: leakySquared / rows.length,
    };
  }

  function drawLeakageChart(svg, values) {
    const width = 760;
    const height = 330;
    const margin = { top: 42, right: 40, bottom: 62, left: 78 };
    const maxValue = Math.max(values.correct, values.leaky) * 1.18;
    const plotHeight = height - margin.top - margin.bottom;
    const y = (value) => height - margin.bottom - (value / maxValue) * plotHeight;
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => maxValue * fraction);
    const bars = [
      { x: 190, value: values.correct, className: "bar-correct", label: "Pipeline correta" },
      { x: 445, value: values.leaky, className: "bar-leaky", label: "Transformação vazada" },
    ];
    svg.innerHTML = `
      <title>Erro de validação com e sem vazamento</title>
      <desc>A barra da transformação vazada é artificialmente menor porque usa as respostas da validação.</desc>
      ${ticks.map((tick) => `<line class="grid-line" x1="${margin.left}" x2="${width - margin.right}" y1="${y(tick)}" y2="${y(tick)}"></line><text x="${margin.left - 12}" y="${y(tick) + 4}" text-anchor="end">${fmt(tick, 0)}</text>`).join("")}
      <line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line>
      <line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
      <text x="20" y="${height / 2}" text-anchor="middle" transform="rotate(-90 20 ${height / 2})">MSE em validação</text>
      ${bars.map((bar) => {
        const top = y(bar.value);
        return `<rect class="${bar.className}" x="${bar.x}" y="${top}" width="125" height="${height - margin.bottom - top}" rx="5"></rect>
          <text x="${bar.x + 62.5}" y="${top - 12}" text-anchor="middle" font-weight="700">${fmt(bar.value, 0)}</text>
          <text x="${bar.x + 62.5}" y="${height - 25}" text-anchor="middle">${bar.label}</text>`;
      }).join("")}`;
  }

  function initializeLeakageLab() {
    if (!$("leakage-lab")) return;
    let seed = 2407;

    function update() {
      const perNeighborhood = Number($("leakage-size").value);
      const values = targetEncodingMse(seed, perNeighborhood);
      const gap = 100 * (values.correct - values.leaky) / values.correct;
      $("leakage-size-value").textContent = perNeighborhood;
      $("leakage-correct-mse").textContent = fmt(values.correct, 0);
      $("leakage-leaky-mse").textContent = fmt(values.leaky, 0);
      $("leakage-gap").textContent = `${fmt(gap, 1)}%`;
      $("leakage-interpretation").textContent = perNeighborhood <= 4
        ? "Com grupos pequenos, cada resposta tem grande influência sobre a média vazada. O erro parece muito menor justamente quando a avaliação é mais frágil."
        : "Com mais observações por bairro, a influência de uma única resposta diminui, mas o procedimento continua incorreto: a validação ainda participa da construção do atributo.";
      drawLeakageChart($("leakage-chart"), values);
    }

    $("leakage-size").addEventListener("input", update);
    $("leakage-resample").addEventListener("click", () => {
      seed += 101;
      update();
    });
    update();
  }

  const cvModels = {
    m1: { label: "M1 — grau 1", values: [0.045, 0.046, 0.043, 0.042, 0.045] },
    m2: { label: "M2 — grau 2", values: [0.040, 0.040, 0.048, 0.040, 0.040] },
    m3: { label: "M3 — grau 3", values: [0.055, 0.052, 0.053, 0.050, 0.055] },
  };

  function cvMean(values) {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }

  function drawCvChart(svg, model, activeFold) {
    const width = 820;
    const height = 350;
    const margin = { top: 36, right: 34, bottom: 58, left: 72 };
    const yMin = 0.035;
    const yMax = 0.06;
    const plotHeight = height - margin.top - margin.bottom;
    const y = (value) => height - margin.bottom - ((value - yMin) / (yMax - yMin)) * plotHeight;
    const ticks = [0.035, 0.04, 0.045, 0.05, 0.055, 0.06];
    const barWidth = 74;
    const gap = 55;
    const firstX = 125;
    const mean = cvMean(model.values);
    svg.innerHTML = `
      <title>MSE em cada parte para ${model.label}</title>
      <desc>Cinco barras mostram os erros de validação; a parte atual aparece em laranja e a linha tracejada mostra a média final.</desc>
      ${ticks.map((tick) => `<line class="grid-line" x1="${margin.left}" x2="${width - margin.right}" y1="${y(tick)}" y2="${y(tick)}"></line><text x="${margin.left - 10}" y="${y(tick) + 4}" text-anchor="end">${fmt(tick, 3)}</text>`).join("")}
      <line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line>
      <line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
      <line class="mean-line" x1="${margin.left}" x2="${width - margin.right}" y1="${y(mean)}" y2="${y(mean)}"></line>
      <text x="${width - margin.right}" y="${y(mean) - 9}" text-anchor="end">média = ${fmt(mean, 4)}</text>
      ${model.values.map((value, index) => {
        const x = firstX + index * (barWidth + gap);
        const top = y(value);
        return `<rect class="mse-bar${index === activeFold ? " active" : ""}" x="${x}" y="${top}" width="${barWidth}" height="${height - margin.bottom - top}" rx="4"></rect>
          <text x="${x + barWidth / 2}" y="${top - 10}" text-anchor="middle">${fmt(value, 3)}</text>
          <text x="${x + barWidth / 2}" y="${height - 27}" text-anchor="middle">parte ${index + 1}</text>`;
      }).join("")}`;
  }

  function initializeCvLab() {
    if (!$("cv-lab")) return;
    let fold = 0;

    function update() {
      const key = $("cv-model").value;
      const model = cvModels[key];
      const partial = cvMean(model.values.slice(0, fold + 1));
      $("cv-fold-strip").innerHTML = Array.from({ length: 5 }, (_, index) => `
        <div class="${index === fold ? "valid" : "train"}"><strong>Parte ${index + 1}</strong><small>${index === fold ? "validação" : "treino"}</small></div>`).join("");
      $("cv-current-fold").textContent = `${fold + 1} de 5`;
      $("cv-current-mse").textContent = fmt(model.values[fold], 3);
      $("cv-partial-mean").textContent = fmt(partial, 4);
      $("cv-previous").disabled = fold === 0;
      $("cv-next").disabled = fold === 4;
      drawCvChart($("cv-chart"), model, fold);

      const means = Object.fromEntries(Object.entries(cvModels).map(([modelKey, item]) => [modelKey, cvMean(item.values)]));
      $("cv-model-summary").innerHTML = Object.entries(cvModels).map(([modelKey, item]) => `
        <div class="${modelKey === "m2" ? "winner" : ""}"><span>${item.label}</span><strong>${fmt(means[modelKey], 4)}</strong></div>`).join("");
      $("cv-interpretation").textContent = fold < 4
        ? `Até aqui, a média de ${model.label} usa apenas ${fold + 1} ${fold === 0 ? "parte" : "partes"}. Continue para observar quanto ela depende das divisões restantes.`
        : `Concluídas as cinco partes, ${model.label} tem MSE médio ${fmt(means[key], 4)}. Comparando as três médias, M2 é escolhido e será reajustado usando todo o conjunto de desenvolvimento.`;
    }

    $("cv-model").addEventListener("change", update);
    $("cv-previous").addEventListener("click", () => {
      fold = Math.max(0, fold - 1);
      update();
    });
    $("cv-next").addEventListener("click", () => {
      fold = Math.min(4, fold + 1);
      update();
    });
    update();
  }

  function initializeSplitLab() {
    if (!$("split-lab")) return;
    const hosts = ["A", "B", "C", "D", "E", "F"];
    const times = [1, 2, 3, 4, 5];

    function assignment(strategy, hostIndex, time) {
      if (strategy === "time") return time >= 4 ? "valid" : "train";
      if (strategy === "group") return hostIndex >= 4 ? "valid" : "train";
      return (hostIndex * 2 + time) % 5 === 2 ? "valid" : "train";
    }

    function update() {
      const strategy = $("split-strategy").value;
      const cells = ["<div class=\"split-header\">anfitrião</div>"];
      times.forEach((time) => cells.push(`<div class="split-header">t${time}</div>`));
      hosts.forEach((host, hostIndex) => {
        cells.push(`<div class="split-host">Host ${host}</div>`);
        times.forEach((time) => {
          const set = assignment(strategy, hostIndex, time);
          cells.push(`<div class="${set}" aria-label="Host ${host}, tempo ${time}, ${set === "train" ? "treino" : "validação"}">${set === "train" ? "T" : "V"}</div>`);
        });
      });
      $("split-grid").innerHTML = cells.join("");

      let overlap = 0;
      hosts.forEach((host, hostIndex) => {
        const sets = new Set(times.map((time) => assignment(strategy, hostIndex, time)));
        if (sets.size > 1) overlap += 1;
      });
      let lookahead = false;
      hosts.forEach((host, hostIndex) => {
        const validTimes = times.filter((time) => assignment(strategy, hostIndex, time) === "valid");
        const trainTimes = times.filter((time) => assignment(strategy, hostIndex, time) === "train");
        if (validTimes.length && trainTimes.some((time) => time > Math.min(...validTimes))) lookahead = true;
      });

      const scenarios = {
        random: {
          question: "nova linha i.i.d.",
          interpretation: "O K-fold aleatório mistura momentos e mantém cada anfitrião nos dois lados. É adequado apenas quando as linhas podem ser tratadas como aproximadamente intercambiáveis.",
        },
        time: {
          question: "período futuro",
          interpretation: "A validação contém somente os momentos finais. Não há informação do futuro no treino, mas os mesmos anfitriões podem aparecer nos dois lados — coerente se eles continuarão ativos.",
        },
        group: {
          question: "novo anfitrião",
          interpretation: "Anfitriões inteiros ficam fora do treino. A divisão mede generalização para entidades novas, sem permitir que padrões específicos do mesmo anfitrião apareçam nos dois lados.",
        },
      };
      $("split-overlap").textContent = `${overlap} de ${hosts.length}`;
      $("split-lookahead").textContent = lookahead ? "sim" : "não";
      $("split-question").textContent = scenarios[strategy].question;
      $("split-interpretation").textContent = scenarios[strategy].interpretation;
    }

    $("split-strategy").addEventListener("change", update);
    update();
  }

  initializeFeatureLab();
  initializeLeakageLab();
  initializeCvLab();
  initializeSplitLab();
})();
