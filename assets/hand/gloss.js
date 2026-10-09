// English sentence -> ASL sign order, by rule. Inlined into the library page and tested with node.
//
// lex: {signs: {GLOSS: ...}, alias: {WORD: [GLOSS, ...]}, tags: {WORD: S|ING|ED|LY|PAST|PL},
//       nouns: [GLOSS, ...], drop: [WORD, ...]}
// toGloss(text, lex, nlp) -> {items, spelled, dropped, swapped, added}. An item is {gloss, word, rep} (rep: a plural
// noun signed twice) or {spell: WORD} for a word with no sign.
// nlp (optional, translate.analyze): spaCy's reading of each word, [{word, lemma, pos, tag, neg}, ...]. With it, a
// word without a sign falls back to its dictionary form (BIGGER = BIG), and noun or verb, past tense and plural
// come from the sentence (my BOOKS, he BOOKS a room) instead of the word list.
//
// Rules, per sentence:
//  1. Words become signs, longest run first (THANK YOU, HARD OF HEARING), then aliases and word forms
//     (WENT = GO). English-only words are dropped: articles, forms of be, the helper DO, TO, AT, OF.
//  2. GOING TO + verb is WILL; HAVE TO / HAS TO / HAD TO is MUST; HAVE + past verb (I have eaten) loses
//     HAVE and is marked past like any other past verb.
//  3. Time comes first: time signs and phrases (YESTERDAY, LAST WEEK, NEXT YEAR, ON MONDAY, TWO DAY AGO)
//     move to the front, without ON/IN/AT. WILL is left out once a time sign says when.
//  4. Past tense with no time sign: FINISH closes the clause (I EAT PIZZA FINISH). A past state with no past
//     verb (I was sick) starts with PAST. Negated or questioned past gets no marker.
//  5. WH-questions end with the question sign (YOUR NAME WHAT, YOU LIVE WHERE, YOU HAVE HOW MANY).
//  6. A plural noun is signed once after a number or quantity sign (THREE DOG, MANY BOOK) and twice
//     otherwise (CAT CAT, shifted sideways). Verbs with -s (WANTS) are signed once.
(function (root) {
  const NUM = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN'];
  const DAYS = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'];
  const UNITS = new Set(['DAY', 'WEEK', 'MONTH', 'YEAR', 'MORNING', 'AFTERNOON', 'EVENING', 'NIGHT', 'WEEKEND',
    'HOUR', 'MINUTE', ...DAYS]);
  const TIME = new Set(['YESTERDAY', 'TODAY', 'TOMORROW', 'NOW', 'TONIGHT', 'LATER', 'SOON', 'RECENT', 'LAST WEEK',
    'LAST YEAR', 'EVERYDAY', 'DAILY', 'MORNING', 'AFTERNOON', 'EVENING', 'WEEKEND', ...DAYS]);
  const TIME_LEAD = new Set(['LAST', 'NEXT', 'THIS', 'EVERY']);  // + a unit: NEXT WEEK, THIS MORNING
  const TIME_PREP = new Set(['ON', 'IN', 'AT']);                 // left out before a time phrase
  const WH = new Set(['WHAT', 'WHERE', 'WHO', 'WHEN', 'WHY', 'HOW', 'WHICH']);
  const QUANT = new Set(['MANY', 'SOME', 'FEW', 'SEVERAL', 'ALL', 'BOTH', 'MORE', 'LOT', 'A LOT', 'MUCH', 'ENOUGH',
    'EVERY', ...NUM]);
  const BE = new Set(['IS', 'AM', 'ARE', 'BE', 'BEEN', 'BEING', 'WAS', 'WERE']);
  const AUX = new Set([...BE, 'DO', 'DOES', 'DID', 'CAN', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'HAVE', 'HAS']);
  const NEG = new Set(['NOT', 'NEVER', 'NONE', 'NOTHING', 'CANNOT', 'REFUSE']);
  const PAST_AUX = new Set(['WAS', 'WERE', 'DID']);

  function words(s) {
    return s.toUpperCase().replace(/[‘’`]/g, "'").replace(/[^A-Z0-9' ]+/g, ' ').split(/\s+/)
      .map(x => x.replace(/^'+|'+$/g, '')).filter(Boolean);
  }

  // spaCy's fine tag -> the word-form tag the rules read
  const NLP_TAG = {VBD: 'PAST', VBN: 'PAST', VBZ: 'S', NNS: 'S', NNPS: 'S', VBG: 'ING'};

  function toGloss(text, lex, nlp) {
    const signs = lex.signs, alias = lex.alias || {}, tags = lex.tags || {}, nouns = new Set(lex.nouns || []);
    const drop = new Set([...(lex.drop || []), 'TO']);
    const has = g => !!signs[g];
    const glossesOf = x => {
      for (const k of [x, x.endsWith("'S") ? x.slice(0, -2) : null]) if (k) {
        if (signs[k]) return [k];
        if (alias[k]) return alias[k];
      }
      return null;
    };
    const maxw = Math.max(1, ...Object.keys(signs).concat(Object.keys(alias)).map(g => g.split(' ').length));
    const out = {items: [], spelled: [], dropped: [], swapped: [], added: []};
    const all = words(text), single = all.length === 1;  // a lone word is always signed or spelled, never dropped
    if (nlp && (nlp.length !== all.length || nlp.some((x, i) => x.word !== all[i]))) nlp = null;  // out of step
    const isNoun = x => x.nlp ? x.nlp.pos === 'NOUN' || x.nlp.pos === 'PROPN' : nouns.has(x.g);
    let wi = 0;
    for (const sentence of text.split(/(?<=[.!?;])\s*/).filter(s => words(s).length)) {
      const question = /\?\s*$/.test(sentence.trim());
      const w = words(sentence), info = nlp ? nlp.slice(wi, wi + w.length) : [];
      wi += w.length;
      // 1. tokens {g, word, tag, kind: sign|num|drop|spell}
      let t = [];
      for (let i = 0; i < w.length;) {
        let n = Math.min(maxw, w.length - i);
        for (; n > 1 && !glossesOf(w.slice(i, i + n).join(' ')); n--);
        const k = w.slice(i, i + n).join(' '), nl = n === 1 ? info[i] || null : null;
        i += n;
        if (n === 1 && drop.has(k) && !single) { t.push({g: null, word: k, kind: 'drop'}); continue; }
        let gs = glossesOf(k), tag = tags[k] || null;
        // no sign for the word as written: try its dictionary form, but never a noun's sign for a verb (FIRED
        // is not the FIRE sign; the verb forms a noun sign does take are already in the alias table)
        const lem = nl && !gs && nl.lemma && nl.lemma !== k ? glossesOf(nl.lemma) : null;
        if (lem && !(nl.pos === 'VERB' && nouns.has(lem[lem.length - 1]))) { gs = lem; tag = NLP_TAG[nl.tag] || null; }
        if (gs) {
          if (gs.join(' ') !== k) out.swapped.push(k.toLowerCase() + ' = ' + gs.join(' '));
          gs.forEach((g, j) => t.push({g, word: k, tag: j === gs.length - 1 ? tag : null, kind: 'sign', nlp: nl}));
        } else if (/^[0-9]+$/.test(k)) {
          (+k <= 10 && has(NUM[+k]) ? [NUM[+k]] : [...k].map(d => NUM[d])).filter(has)
            .forEach(g => t.push({g, word: k, kind: 'num'}));
        } else t.push({g: null, word: k, kind: 'spell'});
      }
      const W = k => t[k] ? (t[k].g || t[k].word) : null;
      // 2. GOING TO + verb -> WILL, HAVE TO -> MUST
      t.forEach((x, k) => {
        const next = t[k + 2];
        if (x.word === 'GOING' && W(k + 1) === 'TO' && next && next.kind === 'sign' && !isNoun(next) && has('WILL'))
          Object.assign(x, {g: 'WILL', tag: null});
        if (['HAVE', 'HAS', 'HAD'].includes(x.word) && W(k + 1) === 'TO' && has('MUST'))
          Object.assign(x, {g: 'MUST', tag: null});
        // HAVE + past verb is English's perfect (I have eaten), not owning: ASL marks it with FINISH alone
        else if (['HAVE', 'HAS', 'HAD'].includes(x.word) && t[k + 1] && !isNoun(t[k + 1])
                 && (t[k + 1].nlp ? t[k + 1].nlp.tag === 'VBN' : ['PAST', 'ED'].includes(t[k + 1].tag)))
          Object.assign(x, {g: null, kind: 'drop'});
      });
      // tense, read from the English before anything moves
      const negated = t.some(x => NEG.has(x.g) || x.nlp && x.nlp.neg);
      let pastVerb = null, pastAux = false;
      t.forEach((x, k) => {
        const afterBe = k > 0 && BE.has(t[k - 1].word);  // I am tired: an adjective, not a past verb
        const past = x.nlp ? x.nlp.pos === 'VERB' && (x.nlp.tag === 'VBD' || x.nlp.tag === 'VBN' && !afterBe)
                           : (x.tag === 'PAST' || x.tag === 'ED' && !afterBe) && !nouns.has(x.g);
        if (!pastVerb && x.kind === 'sign' && past) pastVerb = x;
        if (PAST_AUX.has(x.word)) pastAux = true;
      });
      // 3. time phrases to the front
      const time = [];
      for (let k = 0; k < t.length; k++) {
        const g = W(k);
        let b = -1;
        if (TIME.has(g)) b = k;
        else if (TIME_LEAD.has(g) && UNITS.has(W(k + 1))) b = k + 1;
        else if (t[k].kind === 'num' || QUANT.has(g)) {
          let e = k;
          while (t[e + 1] && t[e + 1].kind === 'num') e++;
          if (UNITS.has(W(e + 1)) && W(e + 2) === 'AGO') b = e + 1;
        }
        if (b < 0) continue;
        if (W(b + 1) === 'AGO') b++;
        let a = k;
        while (a > 0 && (TIME_PREP.has(t[a - 1].word) || t[a - 1].kind === 'drop')) a--;
        const span = t.splice(a, b - a + 1);
        span.filter(x => x.kind === 'drop' || TIME_PREP.has(x.word)).forEach(x => out.dropped.push(x.word));
        time.push(...span.filter(x => x.kind !== 'drop' && !TIME_PREP.has(x.word)));
        k = a - 1;
      }
      t = t.filter(x => {
        if (x.kind === 'drop') out.dropped.push(x.word);
        return x.kind !== 'drop';
      });
      if (time.length) t = t.filter(x => x.g !== 'WILL');
      // 4. past tense marker
      if (!time.length && !negated && !question) {
        if (pastVerb && t.includes(pastVerb) && !t.some(x => x.g === 'FINISH') && has('FINISH')) t.push({g: 'FINISH', word: '', kind: 'sign', added: true});
        else if (!pastVerb && pastAux && has('PAST')) t.unshift({g: 'PAST', word: '', kind: 'sign', added: true});
      }
      t = time.concat(t);
      // 6. plurals (counted in English order, before a WH-sign moves: HOW MANY CAT)
      t.forEach((x, k) => {
        const plural = x.nlp ? ['NNS', 'NNPS'].includes(x.nlp.tag) : (x.tag === 'S' || x.tag === 'PL') && nouns.has(x.g);
        if (x.kind !== 'sign' || !plural) return;
        x.rep = ![t[k - 1], t[k - 2]].some(y => y && (y.kind === 'num' || QUANT.has(y.g)));
      });
      // 5. WH-question: the question sign goes last
      const wh = t.findIndex(x => WH.has(x.g));
      const startsWh = WH.has(w[0]) || time.length && wh === time.length;
      if (wh >= 0 && startsWh && t.length > 1 && (question || AUX.has(w[1]))) {
        const n = t[wh].g === 'HOW' && QUANT.has(W(wh + 1)) ? 2 : 1;  // HOW MANY, HOW MUCH stay together
        t.push(...t.splice(wh, n));
      }
      for (const x of t) {
        if (x.kind === 'spell') { out.items.push({spell: x.word}); out.spelled.push(x.word); continue; }
        out.items.push({gloss: x.g, word: x.word, rep: !!x.rep});
        if (x.added) out.added.push(x.g);
      }
    }
    return out;
  }

  root.toGloss = toGloss;
  if (typeof module !== 'undefined') module.exports = {toGloss};
})(typeof window !== 'undefined' ? window : globalThis);
