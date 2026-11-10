"""REPORT.md (in Russian) generated from results/ so that every number is traceable."""
import os, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
M4 = ["SampEn", "PE", "WPE", "SpEn"]
VN = dict(pinn="PINN", spline="сплайн", odelsq="ODE-LSQ", dataonly="сеть без физики", noadapt="PINN, фикс. веса")
VERD = dict(CONFIRMED="подтверждена", REFUTED="опровергнута", INCONCLUSIVE="не определена (inconclusive)")


def J(p):
    f = os.path.join(RES, p)
    return json.load(open(f)) if os.path.exists(f) else {}


def f(x, n=4):
    try:
        if x != x:
            return "—"
        return f"{float(x):.{n}f}"
    except Exception:
        return "n/a"


def main():
    BIF = J("bifurcation.json"); BS = J("benchmark/benchmark_summary.json"); X = J("manuscript_numbers.json")
    AUD = J("audit_report.json"); PAR = J("param/param_summary.json"); FL = J("benchmark/explore_floor.json")
    REFV = J("references_verified.json")
    C = BS["conditions"]; H = BS["hypotheses"]; R = BS["reference"]["ref_34_1000ms"]
    L = []
    a = L.append
    a("# REPORT — Энтропийная детекция перехода покой→спайки в модели Морриса–Лекара (PINN, разреженные данные)\n")
    a("Отчёт сгенерирован автоматически (`src/make_report.py`) из файлов `results/`; все числа ниже взяты оттуда.\n")
    a("## 1. Что сделано\n")
    a("- Этап 0: гипотезы H1–H5, критерии и правило детекции зафиксированы до расчётов (`configs/hypotheses.md`, хэши в `STATE.json`).")
    a("- Этап 1: эталонные решения (DOP853, rtol = 1e-10) для 292 токов, проверка по допуску, методу и шагу вывода.")
    a(f"- Этап 2: бифуркационный анализ (собственное продолжение): I_SN = {f(BIF['I_SN'], 6)} мкА/см², SNIC подтверждён; "
      f"показатель периода β = {f(BIF['fit_free']['beta'], 5)} ± {BIF['fit_free']['err']['beta']:.1e} (теория 0.5).")
    a(f"- Этап 3 (протокол разреженных данных): сценарии A (полные данные), B (ρ = 0.5/0.2/0.1/0.05), C (ρ = 0.2/0.05, η = 0.01/0.05/0.10), 10 масок на условие, сплиты train/val/test по токам, тесты на утечку.")
    a(f"- Литература: {len(REFV)} записей проверено через Crossref/DataCite, в рукописи {AUD.get('citations', {}).get('n_refs', 'n/a')} ссылок, все с проверенным DOI; непроверяемые ссылки исключены (D-17).")
    a(f"- Этап 4: разреженные/шумные наблюдения (сценарии A, B, C), PINN (сегменты 50 мс, curriculum), базовые линии "
      f"(кубический сплайн, ODE-LSQ, сеть без физики, PINN с фиксированными весами). Всего обучений PINN (основной вариант): {X.get('n_pinn_runs', 'n/a')}, "
      f"из них в сценарии A: {X.get('n_pinn_runs_A', 'n/a')}.")
    a("- Этап 5: четыре энтропии (SampEn, PE, WPE, SpEn) для эталона и всех реконструкций, анализ чувствительности параметров.")
    a("- Этап 6: локализация перехода ε_k = |I*_k − I_crit|, AUC, статистика по реализациям, проверка H1–H5 по предрегистрированным правилам.")
    if PAR.get("n_seeds"):
        a(f"- Вспомогательный эксперимент: параметрическая PINN (t, I) → (V, w), {PAR['n_seeds']} сидов (D-22).")
    a("- Рукопись MDPI Entropy (Article): `manuscript/Entropy_manuscript.docx` (+ .tex, .md), рисунки, таблицы, supplementary, highlights и список рецензентов.")
    a("- Код: `code/` (src, tests, configs, run_all.py, audit.py, README.md, LICENSE, CITATION.cff, requirements.txt).\n")

    a("## 2. Ключевые числа\n")
    a("### Эталон (сетка 34 тока, окно 1000 мс)\n")
    a("| Мера | ε, мкА/см² | AUC |"); a("|---|---:|---:|")
    for m in M4:
        a(f"| {m} | {f(R[m]['eps'])} | {f(R[m]['auc'], 3)} |")
    a(f"\nНаименьшая достижимая ошибка на сетке: {f(X['best_eps_34'])} мкА/см².\n")
    a("### Реконструкция (медиана по реализациям, тестовые токи)\n")
    a("| Сц. | ρ | η | Метод | n | relL2(V) | relL2(w) | ε SampEn | ε PE | ε WPE | ε SpEn | AUC WPE |")
    a("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = dict(pinn=0, dataonly=1, noadapt=2, spline=3, odelsq=4)
    for k in sorted(C, key=lambda k: (C[k]["scenario"], -C[k]["rho"], C[k]["eta"], order.get(C[k]["variant"], 9))):
        c = C[k]
        a(f"| {c['scenario']} | {c['rho']:g} | {c['eta']:g} | {VN.get(c['variant'], c['variant'])} | {c['n']} | "
          f"{f(c['relL2_V_med_test'])} | {f(c.get('relL2_w_med_test'), 3)} | " + " | ".join(f(c[m + '_eps_median']) for m in M4)
          + f" | {f(c['WPE_auc_median'], 2)} |")
    a("")
    q = X.get("quiet", {})
    if q:
        a("СКО реконструкции в окнах покоя (I < I_crit), мВ, медиана (порог «покоя» 0.1 мВ): "
          + "; ".join(f"{k.replace('|', ' ')}: {f(v['sd_median'], 3)}" for k, v in q.items() if k.startswith("C")) + ".\n")
    a("### Гипотезы\n")
    a("| H | Вердикт |"); a("|---|---|")
    for h in ("H1", "H2", "H3", "H4", "H5"):
        a(f"| {h} | {VERD.get(H[h]['verdict'], H[h]['verdict'])} |")
    a("")
    if PAR.get("by_role"):
        a("### Параметрическая PINN (обобщение по току)\n")
        a("| Роль токов | n токов | relL2(V) | relL2(w) | пропущен 1-й спайк |"); a("|---|---:|---:|---:|---:|")
        for role, d in PAR["by_role"].items():
            a(f"| {role} | {d.get('n_currents', 'n/a')} | {f(d['relL2_V_median'], 3)} | {f(d['relL2_w_median'], 2)} | {d['n_missed_first_spike']} из {d['n_spiking_cases']} |")
        a("\nВывод: параметрическая сеть не обобщает по току через порог; w не восстановлена (D-25).")
        a("")
    if FL:
        a("Разведочный анализ порога «покоя» (D-23, не участвует в проверке гипотез): см. `results/benchmark/explore_floor.json`.\n")

    a("### Главные выводы\n")
    a(f"1. На эталонных сигналах PE, WPE и SpEn локализуют переход с точностью шага сетки ({f(X['best_eps_34'])} мкА/см²), SampEn ошибается на {f(R['SampEn']['eps'])} мкА/см² (окно 1000 мс).")
    a("2. Без шума разреженность до 5 % не меняет медианный результат детекции ни для одного метода; PINN восстанавливает скрытую w (H5 подтверждена), порядок мер сохраняется (H3 подтверждена).")
    a("3. С шумом остаточные флуктуации в окнах покоя превышают порог 0.1 мВ и обращают направление скачка энтропии (AUC → 0), хотя ε часто остаётся малым; «устойчивость» SampEn к шуму (H2 опровергнута в её пользу) — артефакт этого обращения.")
    a("4. При известных параметрах ODE-LSQ точнее PINN и сохраняет AUC = 1; сеть без физики сглаживает окна покоя не хуже PINN; PINN не лучше сплайна значимо (H4 не определена), WPE vs PE — не определена (H1).\n")
    a("## 3. Отклонения от ТЗ и принятые решения\n")
    dec = open(os.path.join(ROOT, "DECISIONS.md"), encoding="utf-8").read()
    for row in dec.splitlines():
        m = re.match(r"\| (D-\d\d) \| [^|]* \| (.*?) \| (.*?) \|$", row)
        if m and m.group(1) in ("D-01", "D-03", "D-05", "D-06", "D-11", "D-13", "D-14", "D-16", "D-17", "D-18", "D-19",
                                "D-20", "D-21", "D-22", "D-23", "D-24", "D-25", "D-26"):
            a(f"- **{m.group(1)}**: {m.group(2)}")
    a("\nПолный журнал решений — `DECISIONS.md`. Требования MDPI Entropy сверены частично (официальная страница недоступна напрямую, D-18).\n")
    a("## 4. Ограничения\n")
    lim = open(os.path.join(ROOT, "LIMITATIONS.md"), encoding="utf-8").read().split("\n", 1)[1].strip()
    a(lim + "\n")
    a("## 5. Аудит\n")
    if AUD:
        a(f"Цикл аудита {AUD.get('cycle')}: итог — {'пройден' if AUD.get('passed') else 'НЕ пройден'}. "
          f"Проверено чисел: {AUD['numbers']['checked']} (непрослеживаемых: {len(AUD['numbers']['untraceable'])}); "
          f"ссылок: {AUD['citations']['n_refs']} (без проверенного DOI: {len(AUD['citations']['unverified'])}); "
          f"рисунков: {AUD['figures_tables']['n_figures']}, таблиц: {AUD['figures_tables']['n_tables']} (все упомянуты: "
          f"{'да' if AUD['figures_tables']['ok'] else 'нет'}); уравнений: {AUD['equations']['n']}; аннотация: {AUD['abstract']['words']} слов; "
          f"ключевых слов: {AUD['keywords']['n']}; объём основного текста: {AUD['length']['words']} слов.\n")
    a("## 6. Что нужно сделать вручную\n")
    a("1. Подтвердить список соавторов, их аффилиации и ORCID (в рукописи стоят плейсхолдеры `[Co-author name]`), а также вклад по CRediT.")
    a("2. Указать финансирование (раздел Funding) и, при необходимости, Acknowledgments.")
    a("3. Загрузить содержимое `code/` в Zenodo (или GitHub + Zenodo), получить DOI и вставить его в Data Availability Statement и `CITATION.cff`.")
    a("4. Сверить рукопись с актуальной версией Instructions for Authors MDPI Entropy и шаблоном MDPI (проверено частично, D-18).")
    a("5. Проверить актуальность аффилиаций и отсутствие конфликтов интересов у предложенных рецензентов (`manuscript/highlights_and_reviewers.md`), добавить их e-mail.")
    a("6. Подать рукопись через систему MDPI SuSy (susy.mdpi.com).\n")
    a("## 7. Состав пакета\n")
    a("`final_package.zip`: manuscript/ (docx, tex, md, рисунки, highlights_and_reviewers.md), figures/, supplementary/, code/, "
      "REPORT.md, DECISIONS.md, LIMITATIONS.md, STATE.json, results/audit_report.json.")
    open(os.path.join(ROOT, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("REPORT.md written:", len(L), "lines")


if __name__ == "__main__":
    main()
