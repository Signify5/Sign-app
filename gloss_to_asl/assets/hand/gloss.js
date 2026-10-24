(function (root) {
  const NUM = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN'];
  const DAYS = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'];
  const GREET = new Set(['MORNING', 'AFTERNOON', 'EVENING', 'NIGHT']);
  const UNITS = new Set(['DAY', 'WEEK', 'MONTH', 'YEAR', 'MORNING', 'AFTERNOON', 'EVENING', 'NIGHT', 'WEEKEND',
    'HOUR', 'MINUTE', ...DAYS]);
  const TIME = new Set(['YESTERDAY', 'TODAY', 'TOMORROW', 'NOW', 'TONIGHT', 'LATER', 'SOON', 'RECENT', 'LAST WEEK',
    'LAST YEAR', 'EVERYDAY', 'DAILY', 'MORNING', 'AFTERNOON', 'EVENING', 'WEEKEND', 'ALL DAY', ...DAYS]);
  const TIME_LEAD = new Set(['LAST', 'NEXT', 'THIS', 'EVERY']);
  const TIME_PREP = new Set(['ON', 'IN', 'AT']);
  const WH = new Set(['WHAT', 'WHERE', 'WHO', 'WHEN', 'WHY', 'HOW', 'WHICH']);
  const QUANT = new Set(['MANY', 'SOME', 'FEW', 'SEVERAL', 'ALL', 'BOTH', 'MORE', 'LOT', 'A LOT', 'MUCH', 'ENOUGH',
    'EVERY', ...NUM]);
  const BE = new Set(['IS', 'AM', 'ARE', 'BE', 'BEEN', 'BEING', 'WAS', 'WERE']);
  const AUX = new Set([...BE, 'DO', 'DOES', 'DID', 'CAN', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'HAVE', 'HAS']);
  const NEG = new Set(['NOT', 'NEVER', 'NONE', 'NOTHING', 'CANNOT', 'REFUSE']);
  const PAST_AUX = new Set(['WAS', 'WERE', 'DID']);

  /* PHRASES_START */
  const PHRASES = [{"en":["HOW","ARE","YOU","DOING"],"gloss":["HOW","YOU"],"id":"how_are_you_doing"},{"en":["NICE","TO","MEET","YOU"],"gloss":["NICE","MEET","YOU"],"id":"nice_to_meet_you"},{"en":["WHAT","IS","YOUR","NAME"],"gloss":["YOUR","NAME","WHAT"],"id":"what_is_your_name"},{"en":["WHERE","DO","YOU","LIVE"],"gloss":["YOU","LIVE","WHERE"],"id":"where_do_you_live"},{"en":["HOW","OLD","ARE","YOU"],"gloss":["YOU","OLD","HOW"],"id":"how_old_are_you"},{"en":["LONG","TIME","NO","SEE"],"gloss":["LONG","TIME","SEE","NOT"],"id":"long_time_no_see"},{"en":["WHAT","TIME","IS","IT"],"gloss":["TIME","WHAT"],"id":"what_time_is_it"},{"en":["HOW","MUCH","IS","IT"],"gloss":["HOW","MUCH"],"id":"how_much_is_it"},{"en":["I","HAVE","TO","GO"],"gloss":["I","MUST","GO"],"id":"i_have_to_go"},{"en":["I","NEED","TO","GO"],"gloss":["I","NEED","GO"],"id":"i_need_to_go"},{"en":["GOING","TO","THE","STORE"],"gloss":["WILL","GO","STORE"],"id":"going_to_store"},{"en":["AS","SOON","AS","POSSIBLE"],"gloss":["SOON","POSSIBLE"],"id":"as_soon_as_possible"},{"en":["ONCE","UPON","A","TIME"],"gloss":["PAST","ONCE"],"id":"once_upon_a_time"},{"en":["HOW","ARE","YOU"],"gloss":["HOW","YOU"],"id":"how_are_you"},{"en":["THANKS","A","LOT"],"gloss":["THANK YOU","A LOT"],"id":"thanks_a_lot"},{"en":["YOU","ARE","WELCOME"],"gloss":["WELCOME"],"id":"you_are_welcome"},{"en":["SEE","YOU","LATER"],"gloss":["SEE","YOU","LATER"],"id":"see_you_later"},{"en":["SEE","YOU","TOMORROW"],"gloss":["TOMORROW","SEE","YOU"],"id":"see_you_tomorrow"},{"en":["WHAT'S","YOUR","NAME"],"gloss":["YOUR","NAME","WHAT"],"id":"whats_your_name"},{"en":["I","LOVE","YOU"],"gloss":["I","LOVE","YOU"],"id":"i_love_you"},{"en":["I","DON'T","KNOW"],"gloss":["I","KNOW","NOT"],"id":"i_dont_know"},{"en":["I","DON'T","UNDERSTAND"],"gloss":["I","UNDERSTAND","NOT"],"id":"i_dont_understand"},{"en":["I","DON'T","WANT"],"gloss":["I","DON'T WANT"],"id":"i_dont_want"},{"en":["PLEASE","HELP","ME"],"gloss":["HELP","I","PLEASE"],"id":"please_help_me"},{"en":["I","AM","HUNGRY"],"gloss":["I","HUNGRY"],"id":"i_am_hungry"},{"en":["I","AM","TIRED"],"gloss":["I","TIRED"],"id":"i_am_tired"},{"en":["I","AM","FINE"],"gloss":["I","FINE"],"id":"i_am_fine"},{"en":["ARE","YOU","OK"],"gloss":["YOU","OK"],"id":"are_you_ok"},{"en":["ARE","YOU","READY"],"gloss":["YOU","READY"],"id":"are_you_ready"},{"en":["WAIT","A","MINUTE"],"gloss":["WAIT","MINUTE"],"id":"wait_a_minute"},{"en":["JUST","A","MINUTE"],"gloss":["WAIT","MINUTE"],"id":"just_a_minute"},{"en":["WANT","TO","EAT"],"gloss":["WANT","EAT"],"id":"want_to_eat"},{"en":["LIKE","TO","SWIM"],"gloss":["LIKE","SWIM"],"id":"like_to_swim"},{"en":["GOING","TO","RAIN"],"gloss":["WILL","RAIN"],"id":"going_to_rain"},{"en":["HAPPY","NEW","YEAR"],"gloss":["HAPPY","NEW","YEAR"],"id":"happy_new_year"},{"en":["DON'T","GIVE","UP"],"gloss":["GIVE UP","NOT"],"id":"give_up"},{"en":["FALL","IN","LOVE"],"gloss":["FALL IN LOVE"],"id":"fall_in_love"},{"en":["HARD","OF","HEARING"],"gloss":["HARD OF HEARING"],"id":"hard_of_hearing"},{"en":["FROM","NOW","ON"],"gloss":["FROM NOW ON"],"id":"from_now_on"},{"en":["A","LITTLE","BIT"],"gloss":["LITTLE BIT"],"id":"little_bit"},{"en":["OF","COURSE","NOT"],"gloss":["SURE","NOT"],"id":"of_course_not"},{"en":["I","THINK","SO"],"gloss":["I","THINK"],"id":"i_think_so"},{"en":["I","HOPE","SO"],"gloss":["I","HOPE"],"id":"i_hope_so"},{"en":["I","MISS","YOU"],"gloss":["I","MISS","YOU"],"id":"i_miss_you"},{"en":["BY","THE","WAY"],"gloss":[],"id":"by_the_way"},{"en":["TWO","DAYS","AGO"],"gloss":["TWO","DAY","AGO"],"id":"two_days_ago"},{"en":["THREE","WEEKS","AGO"],"gloss":["THREE","WEEK","AGO"],"id":"three_weeks_ago"},{"en":["ONE","YEAR","AGO"],"gloss":["ONE","YEAR","AGO"],"id":"one_year_ago"},{"en":["THANK","YOU"],"gloss":["THANK YOU"],"id":"thank_you"},{"en":["GOOD","MORNING"],"gloss":["GOOD","MORNING"],"id":"good_morning"},{"en":["GOOD","AFTERNOON"],"gloss":["GOOD","AFTERNOON"],"id":"good_afternoon"},{"en":["GOOD","EVENING"],"gloss":["GOOD","EVENING"],"id":"good_evening"},{"en":["GOOD","NIGHT"],"gloss":["GOOD","NIGHT"],"id":"good_night"},{"en":["EXCUSE","ME"],"gloss":["EXCUSE"],"id":"excuse_me"},{"en":["I'M","SORRY"],"gloss":["I","SORRY"],"id":"im_sorry"},{"en":["NO","PROBLEM"],"gloss":["PROBLEM","NONE"],"id":"no_problem"},{"en":["OF","COURSE"],"gloss":["SURE"],"id":"of_course"},{"en":["TAKE","CARE"],"gloss":["CARE"],"id":"take_care"},{"en":["HOW","MANY"],"gloss":["HOW","MANY"],"id":"how_many"},{"en":["LET'S","GO"],"gloss":["GO"],"id":"lets_go"},{"en":["COME","HERE"],"gloss":["COME HERE"],"id":"come_here"},{"en":["HAPPY","BIRTHDAY"],"gloss":["HAPPY","BIRTHDAY"],"id":"happy_birthday"},{"en":["MERRY","CHRISTMAS"],"gloss":["HAPPY","CHRISTMAS"],"id":"merry_christmas"},{"en":["WAKE","UP"],"gloss":["WAKE UP"],"id":"wake_up"},{"en":["GET","UP"],"gloss":["GET UP"],"id":"get_up"},{"en":["LOOK","FOR"],"gloss":["LOOK FOR"],"id":"look_for"},{"en":["LOOK","AT"],"gloss":["LOOK AT"],"id":"look_at"},{"en":["PAY","ATTENTION"],"gloss":["PAY ATTENTION"],"id":"pay_attention"},{"en":["SIGN","LANGUAGE"],"gloss":["SIGN LANGUAGE"],"id":"sign_language"},{"en":["ICE","CREAM"],"gloss":["ICE CREAM"],"id":"ice_cream"},{"en":["HOT","DOG"],"gloss":["HOT DOG"],"id":"hot_dog"},{"en":["FRENCH","FRIES"],"gloss":["FRENCH FRIES"],"id":"french_fries"},{"en":["PEANUT","BUTTER"],"gloss":["PEANUT BUTTER"],"id":"peanut_butter"},{"en":["LAST","WEEK"],"gloss":["LAST WEEK"],"id":"last_week"},{"en":["LAST","YEAR"],"gloss":["LAST YEAR"],"id":"last_year"},{"en":["NEXT","WEEK"],"gloss":["NEXT","WEEK"],"id":"next_week"},{"en":["NEXT","YEAR"],"gloss":["NEXT","YEAR"],"id":"next_year"},{"en":["THIS","MORNING"],"gloss":["THIS","MORNING"],"id":"this_morning"},{"en":["THIS","AFTERNOON"],"gloss":["THIS","AFTERNOON"],"id":"this_afternoon"},{"en":["THIS","EVENING"],"gloss":["THIS","EVENING"],"id":"this_evening"},{"en":["EVERY","DAY"],"gloss":["EVERYDAY"],"id":"every_day"},{"en":["ALL","DAY"],"gloss":["ALL DAY"],"id":"all_day"},{"en":["GIVE","UP"],"gloss":["GIVE UP"],"id":"give_up_bare"},{"en":["NOT","YET"],"gloss":["NOT YET"],"id":"not_yet"},{"en":["A","LOT"],"gloss":["A LOT"],"id":"a_lot"},{"en":["RIGHT","NOW"],"gloss":["NOW"],"id":"right_now"},{"en":["ME","TOO"],"gloss":["I","SAME"],"id":"me_too"},{"en":["SAME","HERE"],"gloss":["SAME"],"id":"same_here"},{"en":["WHAT","FOR"],"gloss":["FOR","WHAT"],"id":"what_for"},{"en":["WHO","CARES"],"gloss":["CARE","WHO"],"id":"who_cares"},{"en":["NEVER","MIND"],"gloss":["NEVER","MIND"],"id":"never_mind"},{"en":["HURRY","UP"],"gloss":["HURRY"],"id":"hurry_up"},{"en":["CALM","DOWN"],"gloss":["CALM"],"id":"calm_down"},{"en":["SIT","DOWN"],"gloss":["SIT"],"id":"sit_down"},{"en":["STAND","UP"],"gloss":["STAND"],"id":"stand_up"},{"en":["SHUT","UP"],"gloss":["QUIET"],"id":"shut_up"},{"en":["BE","QUIET"],"gloss":["QUIET"],"id":"be_quiet"},{"en":["CALL","ME"],"gloss":["CALL","I"],"id":"call_me"},{"en":["TEXT","ME"],"gloss":["TEXT","I"],"id":"text_me"},{"en":["TELL","ME"],"gloss":["TELL","I"],"id":"tell_me"},{"en":["SHOW","ME"],"gloss":["SHOW","I"],"id":"show_me"},{"en":["HELP","YOURSELF"],"gloss":["HELP","YOURSELF"],"id":"help_yourself"},{"en":["MAKES","SENSE"],"gloss":["UNDERSTAND"],"id":"make_sense"},{"en":["MAKE","SENSE"],"gloss":["UNDERSTAND"],"id":"make_sense_bare"},{"en":["IN","FACT"],"gloss":["TRUE"],"id":"in_fact"},{"en":["FOR","EXAMPLE"],"gloss":["EXAMPLE"],"id":"for_example"},{"en":["KIND","OF"],"gloss":["SOMETIMES"],"id":"kind_of"},{"en":["A","FEW"],"gloss":["FEW"],"id":"a_few"}];
  /* PHRASES_END */

  function applyPhrases(w) {
    const out = [];
    let i = 0;
    while (i < w.length) {
      let hit = null;
      for (const p of PHRASES) {
        const n = p.en.length;
        if (n && i + n <= w.length && w.slice(i, i + n).every((x, k) => x === p.en[k])) {
          hit = p; break;
        }
      }
      if (hit) { out.push(...hit.gloss); i += hit.en.length; }
      else { out.push(w[i]); i++; }
    }
    return out.filter(Boolean);
  }


  function words(s) {
    return s.toUpperCase().replace(/[''`]/g, "'").replace(/[^A-Z0-9' ]+/g, ' ').split(/\s+/)
      .map(x => x.replace(/^'+|'+$/g, '')).filter(Boolean);
  }

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
    const all = words(text), single = all.length === 1;
    if (nlp && (nlp.length !== all.length || nlp.some((x, i) => x.word !== all[i]))) nlp = null;
    const isNoun = x => x.nlp ? x.nlp.pos === 'NOUN' || x.nlp.pos === 'PROPN' : nouns.has(x.g);
    let wi = 0, si = 0;
    for (const sentence of text.split(/(?<=[.!?;])\s*/).filter(s => words(s).length)) {
      const question = /\?\s*$/.test(sentence.trim());
      const raw = words(sentence), w = applyPhrases(raw), noPast = question || WH.has(raw[0]) || AUX.has(raw[0]), info = (nlp && w.length === raw.length) ? nlp.slice(wi, wi + raw.length) : [];
      wi += raw.length;
      let t = [];
      for (let i = 0; i < w.length;) {
        let n = Math.min(maxw, w.length - i);
        for (; n > 1 && !glossesOf(w.slice(i, i + n).join(' ')); n--);
        const k = w.slice(i, i + n).join(' '), nl = n === 1 ? info[i] || null : null;
        i += n;
        if (n === 1 && drop.has(k) && !single) { t.push({g: null, word: k, kind: 'drop'}); continue; }
        let gs = glossesOf(k), tag = tags[k] || null;
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
      t.forEach((x, k) => {
        const next = t[k + 2];
        if (x.word === 'GOING' && W(k + 1) === 'TO' && next && next.kind === 'sign' && !isNoun(next) && has('WILL'))
          Object.assign(x, {g: 'WILL', tag: null});
        if (['HAVE', 'HAS', 'HAD'].includes(x.word) && W(k + 1) === 'TO' && has('MUST'))
          Object.assign(x, {g: 'MUST', tag: null});
        else if (['HAVE', 'HAS', 'HAD'].includes(x.word) && t[k + 1] && !isNoun(t[k + 1])
                 && (t[k + 1].nlp ? t[k + 1].nlp.tag === 'VBN' : ['PAST', 'ED'].includes(t[k + 1].tag)))
          Object.assign(x, {g: null, kind: 'drop'});
      });
      const negated = t.some(x => NEG.has(x.g) || x.nlp && x.nlp.neg);
      let pastVerb = null, pastAux = false;
      t.forEach((x, k) => {
        const afterBe = k > 0 && BE.has(t[k - 1].word);
        const past = x.nlp ? x.nlp.pos === 'VERB' && (x.nlp.tag === 'VBD' || x.nlp.tag === 'VBN' && !afterBe)
                           : (x.tag === 'PAST' && !afterBe || x.tag === 'ED' && !afterBe) && !nouns.has(x.g);
        if (!pastVerb && x.kind === 'sign' && past) pastVerb = x;
        if (PAST_AUX.has(x.word)) pastAux = true;
      });
      const time = [];
      for (let k = 0; k < t.length; k++) {
        const g = W(k);
        let b = -1;
        if (TIME.has(g)) { if (W(k - 1) === 'GOOD' && GREET.has(g)) continue; b = k; }
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
      if (!time.length && !negated && !noPast) {
        if (pastVerb && t.includes(pastVerb) && !t.some(x => x.g === 'FINISH') && has('FINISH')) t.push({g: 'FINISH', word: '', kind: 'sign', added: true});
        else if (!pastVerb && pastAux && has('PAST')) t.unshift({g: 'PAST', word: '', kind: 'sign', added: true});
      }
      t = time.concat(t);
      t.forEach((x, k) => {
        const plural = x.nlp ? ['NNS', 'NNPS'].includes(x.nlp.tag) : (x.tag === 'S' || x.tag === 'PL') && nouns.has(x.g);
        if (x.kind !== 'sign' || !plural) return;
        x.rep = ![t[k - 1], t[k - 2]].some(y => y && (y.kind === 'num' || QUANT.has(y.g)));
      });
      const negAt = t.findIndex(x => x.g === 'NOT' || x.g === 'NEVER');
      if (negAt >= 0 && negAt < t.length - 1 && !t.some(x => WH.has(x.g))) {
        t.push(...t.splice(negAt, 1));
      }
      const wh = t.findIndex(x => WH.has(x.g));
      const startsWh = WH.has(raw[0]) || time.length && wh === time.length;
      const greeting = raw.length === 3 && raw[0] === 'HOW' && raw[1] === 'ARE' && raw[2] === 'YOU';
      const shaped = !greeting && AUX.has(raw[1]) || raw[0] === 'HOW' && QUANT.has(raw[1]);
      if (wh >= 0 && startsWh && t.length > 1 && !greeting && (question || shaped)) {
        const n = t[wh].g === 'HOW' && QUANT.has(W(wh + 1)) ? 2 : 1;
        t.push(...t.splice(wh, n));
      }
      const hasWh = t.some(x => WH.has(x.g));
      const mark = hasWh && (question || startsWh || shaped) ? 'wh'
        : t.length > 1 && (question || AUX.has(raw[0]) && !hasWh) ? 'yn' : null;
      const sent = si++;
      for (const x of t) {
        if (x.kind === 'spell') { out.items.push({spell: x.word, mark, sent}); out.spelled.push(x.word); continue; }
        out.items.push({gloss: x.g, word: x.word, rep: !!x.rep, mark, sent, neg: NEG.has(x.g) || x.g === 'NO', yes: x.g === 'YES'});
        if (x.added) out.added.push(x.g);
      }
    }
    return out;
  }

  root.toGloss = toGloss;
  if (typeof module !== 'undefined') module.exports = {toGloss};
})(typeof window !== 'undefined' ? window : globalThis);
