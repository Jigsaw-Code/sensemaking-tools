import { SupportedLanguage } from "./languages";

// Statistics-related messages that appear in reports
export const STATISTICS_MESSAGES = {
  statements: {
    "en": "statements",
    "zh-TW": "個意見",
    "zh-CN": "个意见",
    "fr": "déclarations",
    "es": "declaraciones",
    "ja": "個の声明"
  },
  noCommonGround: {
    "en": `No statements met the thresholds necessary to be considered as a point of common ground (at least {minVoteCount} votes, and at least {minCommonGroundProb} agreement{acrossGroups}).`,
    "zh-TW": `沒有意見達到被視為共同點的必要門檻（至少需要 {minVoteCount} 個投票，且至少需要 {minCommonGroundProb} 的同意率{acrossGroups}）。`,
    "zh-CN": `没有意见达到被视为共同点的必要门槛（至少需要 {minVoteCount} 个投票，且至少需要 {minCommonGroundProb} 的同意率{acrossGroups}）。`,
    "fr": `Aucune déclaration n'a atteint les seuils nécessaires pour être considérée comme un terrain d'entente (au moins {minVoteCount} votes et au moins {minCommonGroundProb} d'accord{acrossGroups}).`,
    "es": `Ninguna declaración cumplió con los umbrales necesarios para ser considerada como un punto de terreno común (al menos {minVoteCount} votos, y al menos {minCommonGroundProb} de acuerdo{acrossGroups}).`,
    "ja": `声明は、共通点と見なすために必要な閾値を満たしていません（少なくとも{minVoteCount}票、かつ少なくとも{minCommonGroundProb}の同意率{acrossGroups}）。`
  },
  noDifferencesOfOpinion: {
    "en": `No statements met the thresholds necessary to be considered as a significant difference of opinion (at least {minVoteCount} votes, and more than {minAgreeProbDifference} difference in agreement rate between groups).`,
    "zh-TW": `沒有意見達到被視為顯著意見分歧的必要門檻（至少需要 {minVoteCount} 個投票，且群組間同意率差異超過 {minAgreeProbDifference}）。`,
    "zh-CN": `没有意见达到被视为显著意见分歧的必要门槛（至少需要 {minVoteCount} 个投票，且群组间同意率差异超过 {minAgreeProbDifference}）。`,
    "fr": `Aucune déclaration n'a atteint les seuils nécessaires pour être considérée comme une différence d'opinion significative (au moins {minVoteCount} votes et plus de {minAgreeProbDifference} de différence dans le taux d'accord entre les groupes).`,
    "es": `Ninguna declaración cumplió con los umbrales necesarios para ser considerada como una diferencia de opinión significativa (al menos {minVoteCount} votos, y más de {minAgreeProbDifference} de diferencia en la tasa de acuerdo entre grupos).`,
    "ja": `声明は、意見の相違として見なすために必要な閾値を満たしていません（少なくとも{minVoteCount}票、かつグループ間の同意率の差が{minAgreeProbDifference}を超える）。`
  },
  noCommonGroundDisagree: {
    "en": `No statements met the thresholds necessary to be considered as a point of common ground (at least {minVoteCount} votes, and at least {minCommonGroundProb} agreement across groups).`,
    "zh-TW": `沒有意見達到被視為群組間共同點的必要門檻（至少需要 {minVoteCount} 個投票，且至少需要 {minCommonGroundProb} 的群組間同意率）。`,
    "zh-CN": `没有意见达到被视为群组间共同点的必要门槛（至少需要 {minVoteCount} 个投票，且至少需要 {minCommonGroundProb} 的群组间同意率）。`,
    "fr": `Aucune déclaration n'a atteint les seuils nécessaires pour être considérée comme un terrain d'entente entre les groupes (au moins {minVoteCount} votes et au moins {minCommonGroundProb} d'accord entre les groupes).`,
    "es": `Ninguna declaración cumplió con los umbrales necesarios para ser considerada como un punto de terreno común entre grupos (al menos {minVoteCount} votos, y al menos {minCommonGroundProb} de acuerdo entre grupos).`,
    "ja": `声明は、グループ間の共通点と見なすために必要な閾値を満たしていません（少なくとも{minVoteCount}票、かつグループ間の同意率が少なくとも{minCommonGroundProb}）。`
  },
  noDifferencesOfOpinionGroups: {
    "en": `No statements met the thresholds necessary to be considered as a significant difference of opinion (at least {minVoteCount} votes, and more than {minAgreeProbDifference} difference in agreement rate between groups).`,
    "zh-TW": `沒有意見達到被視為群組間顯著意見分歧的必要門檻（至少需要 {minVoteCount} 個投票，且群組間同意率差異超過 {minAgreeProbDifference}）。`,
    "zh-CN": `没有意见达到被视为群组间显著意见分歧的必要门槛（至少需要 {minVoteCount} 个投票，且群组间同意率差异超过 {minAgreeProbDifference}）。`,
    "fr": `Aucune déclaration n'a atteint les seuils nécessaires pour être considérée comme une différence d'opinion significative entre les groupes (au moins {minVoteCount} votes et plus de {minAgreeProbDifference} de différence dans le taux d'accord entre les groupes).`,
    "es": `Ninguna declaración cumplió con los umbrales necesarios para ser considerada como una diferencia de opinión significativa entre grupos (al menos {minVoteCount} votos, y más de {minAgreeProbDifference} de diferencia en la tasa de acuerdo entre grupos).`,
    "ja": `声明は、グループ間の意見の相違として見なすために必要な閾値を満たしていません（少なくとも{minVoteCount}票、かつグループ間の同意率の差が{minAgreeProbDifference}を超える）。`
  }
};

export function getStatisticsMessage(
  messageType: keyof typeof STATISTICS_MESSAGES,
  lang: SupportedLanguage,
  replacements: Record<string, string | number>
): string {
  const message = STATISTICS_MESSAGES[messageType][lang] || STATISTICS_MESSAGES[messageType]["en"];
  let text = message;
  
  if (replacements) {
    Object.entries(replacements).forEach(([key, value]) => {
      text = text.replace(new RegExp(`{${key}}`, 'g'), value.toString());
    });
  }
  
  return text;
}
