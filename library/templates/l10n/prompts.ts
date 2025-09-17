// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Multi-language prompts for different summarization tasks

import { SupportedLanguage } from "./languages";

/**
 * Multi-language prompt for generating prominent themes
 */
export const THEMES_PROMPT: Record<SupportedLanguage, string> = {
  "en": `Please write a concise bulleted list identifying up to 5 prominent themes across all statements. These statements are all about {topicName}. For each theme, begin with a short theme description written in bold text, followed by a colon, then followed by a SINGLE sentence explaining the theme. Your list should meet the below Criteria and STRICTLY follow the Output Format. Do not preface the bulleted list with any text.

<criteria format="markdown">
* Impartiality: Do not express your own opinion or pass normative judgments on the statements, like agreement, disagreement, or alarm.
* Faithfulness: Your list should accurately reflect the statements without hallucinations or mischaracterizations.
  * Similarly, your list should not assume or misstate the amount of agreement across statements. For example, do not present a theme as unanimous if it is only mentioned in some statements.
  * This criterion also applies to the name of the theme itself: do not assume overwhelming agreement when you name themes if it does not exist. For example, do not name a theme "Support for _______" unless there is overwhelming evidence beyond a reasonable doubt in the statements.
  * Be **specific**. Avoid overgeneralizations or fuzzy nouns like "things" or "aspects".
* Comprehensiveness: Your list should reflect ALL opinions proportional to their representation in the statements. However, **absolutely do not exclude minority opinions**, especially if there are strong objections or mixed stances. Please be **specific** in including these objections or stances.
* Consistent terminology: You should always use "statements" and NOT "comments".
</criteria>

<output_format format="markdown">
* **Title Case Theme**: Sentence
</output_format>`,

  "zh-TW": `請撰寫一個簡潔的項目符號清單，識別所有陳述中最多5個突出主題。這些陳述都是關於{topicName}的。對於每個主題，請以粗體文字開始簡短的主題描述，後接冒號，然後是解釋該主題的單一句子。您的清單應符合以下標準並嚴格遵循輸出格式。請勿在項目符號清單前添加任何文字。

<criteria format="markdown">
* 公正性：請勿表達您自己的意見或對陳述做出規範性判斷，如同意、不同意或警報。
* 忠實性：您的清單應準確反映陳述，不得有幻覺或錯誤描述。
  * 同樣地，您的清單不應假設或錯誤陳述陳述間的一致程度。例如，如果某個主題僅在某些陳述中被提及，請勿將其呈現為一致同意。
  * 此標準也適用於主題本身的命名：如果您沒有壓倒性證據，請勿假設壓倒性同意來命名主題。例如，除非陳述中有壓倒性證據，否則請勿將主題命名為「支持_______」。
  * 要具體。避免過度概括或模糊名詞如「事物」或「方面」。
* 全面性：您的清單應按比例反映陳述中所有意見的代表性。但是，絕對不要排除少數意見，特別是有強烈反對或混合立場的情況。請具體包含這些反對或立場。
* 一致術語：您應始終使用「陳述」和NOT「評論」。
</criteria>

<output_format format="markdown">
* **標題格式主題**：句子
</output_format>`,

  "zh-CN": `请撰写一个简洁的项目符号清单，识别所有陈述中最多5个突出主题。这些陈述都是关于{topicName}的。对于每个主题，请以粗体文字开始简短的主题描述，后接冒号，然后是解释该主题的单个句子。您的清单应符合以下标准并严格遵循输出格式。请勿在项目符号清单前添加任何文字。

<criteria format="markdown">
* 公正性：请勿表达您自己的意见或对陈述做出规范性判断，如同意、不同意或警报。
* 忠实性：您的清单应准确反映陈述，不得有幻觉或错误描述。
  * 同样地，您的清单不应假设或错误陈述陈述间的一致程度。例如，如果某个主题仅在某些陈述中被提及，请勿将其呈现为一致同意。
  * 此标准也适用于主题本身的命名：如果您没有压倒性证据，请勿假设压倒性同意来命名主题。例如，除非陈述中有压倒性证据，否则请勿将主题命名为「支持_______」。
  * 要具体。避免过度概括或模糊名词如「事物」或「方面」。
* 全面性：您的清单应按比例反映陈述中所有意见的代表性。但是，绝对不要排除少数意见，特别是有强烈反对或混合立场的情况。请具体包含这些反对或立场。
* 一致术语：您应始终使用「陈述」而非「评论」。
</criteria>

<output_format format="markdown">
* **标题格式主题**：句子
</output_format>`,

  "fr": `Veuillez rédiger une liste concise à puces identifiant jusqu'à 5 thèmes prédominants à travers toutes les déclarations. Ces déclarations concernent toutes {topicName}. Pour chaque thème, commencez par une brève description du thème écrite en texte gras, suivie de deux points, puis d'une SEULE phrase expliquant le thème. Votre liste doit respecter les critères ci-dessous et suivre STRICTEMENT le format de sortie. N'introduisez pas la liste à puces par aucun texte.

<criteria format="markdown">
* Impartialité : N'exprimez pas votre propre opinion ou ne portez pas de jugements normatifs sur les déclarations, comme l'accord, le désaccord ou l'alarme.
* Fidélité : Votre liste doit refléter fidèlement les déclarations sans hallucinations ou caractérisations erronées.
  * De même, votre liste ne doit pas supposer ou mal déclarer le degré d'accord entre les déclarations. Par exemple, ne présentez pas un thème comme unanime s'il n'est mentionné que dans certaines déclarations.
  * Ce critère s'applique également au nom du thème lui-même : ne supposez pas un accord écrasant lorsque vous nommez des thèmes s'il n'existe pas. Par exemple, ne nommez pas un thème "Soutien pour _______" sauf s'il y a des preuves écrasantes au-delà de tout doute raisonnable dans les déclarations.
  * Soyez **spécifique**. Évitez les généralisations excessives ou les noms vagues comme "choses" ou "aspects".
* Exhaustivité : Votre liste doit refléter TOUTES les opinions proportionnellement à leur représentation dans les déclarations. Cependant, **n'excluez absolument pas les opinions minoritaires**, surtout s'il y a de fortes objections ou des positions mixtes. Veuillez être **spécifique** en incluant ces objections ou positions.
* Terminologie cohérente : Vous devez toujours utiliser "déclarations" et NON "commentaires".
</criteria>

<output_format format="markdown">
* **Thème en Titre de Cas** : Phrase
</output_format>`,

  "es": `Por favor, escriba una lista concisa con viñetas que identifique hasta 5 temas prominentes en todas las declaraciones. Estas declaraciones son todas sobre {topicName}. Para cada tema, comience con una breve descripción del tema escrita en texto en negrita, seguida de dos puntos, luego de una SOLA oración explicando el tema. Su lista debe cumplir con los criterios a continuación y seguir ESTRICTAMENTE el formato de salida. No introduzca la lista con viñetas con ningún texto.

<criteria format="markdown">
* Imparcialidad: No exprese su propia opinión o pase juicios normativos sobre las declaraciones, como acuerdo, desacuerdo o alarma.
* Fidelidad: Su lista debe reflejar con precisión las declaraciones sin alucinaciones o caracterizaciones erróneas.
  * Del mismo modo, su lista no debe asumir o declarar incorrectamente la cantidad de acuerdo entre las declaraciones. Por ejemplo, no presente un tema como unánime si solo se menciona en algunas declaraciones.
  * Este criterio también se aplica al nombre del tema en sí: no asuma un acuerdo abrumador cuando nombre temas si no existe. Por ejemplo, no nombre un tema "Apoyo para _______" a menos que haya evidencia abrumadora más allá de toda duda razonable en las declaraciones.
  * Sea **específico**. Evite generalizaciones excesivas o sustantivos vagos como "cosas" o "aspectos".
* Exhaustividad: Su lista debe reflejar TODAS las opiniones proporcionalmente a su representación en las declaraciones. Sin embargo, **absolutamente no excluya las opiniones minoritarias**, especialmente si hay fuertes objeciones o posturas mixtas. Por favor, sea **específico** al incluir estas objeciones o posturas.
* Terminología consistente: Siempre debe usar "declaraciones" y NO "comentarios".
</criteria>

<output_format format="markdown">
* **Tema en Título de Caso**: Oración
</output_format>`,

  "ja": `すべての声明文から最大5つの顕著なテーマを特定する簡潔な箇条書きリストを作成してください。これらの声明文はすべて{topicName}に関するものです。各テーマについて、太字で書かれた短いテーマ説明で始め、コロンに続き、テーマを説明する単一の文で終わってください。あなたのリストは以下の基準を満たし、出力形式を厳密に従う必要があります。箇条書きリストの前にテキストを付けないでください。

<criteria format="markdown">
* 公平性：あなた自身の意見を表現したり、同意、不同意、警報などの声明文について規範的判断を下したりしないでください。
* 忠実性：あなたのリストは、幻覚や誤った特徴付けなしに声明文を正確に反映する必要があります。
  * 同様に、あなたのリストは声明文間の同意の程度を仮定したり誤って述べたりすべきではありません。例えば、テーマが一部の声明文でのみ言及されている場合、それを全会一致として提示しないでください。
  * この基準はテーマ自体の名前にも適用されます：存在しない場合、テーマに名前を付ける際に圧倒的な同意を仮定しないでください。例えば、声明文に合理的な疑いを超える圧倒的な証拠がない限り、テーマを「_______への支持」と名付けないでください。
  * 具体的にしてください。「もの」や「側面」などの過度な一般化や曖昧な名詞を避けてください。
* 包括性：あなたのリストは、声明文での表現に比例してすべての意見を反映する必要があります。しかし、少数意見を絶対に除外しないでください。特別に強い反対や混合した立場がある場合はそうです。これらの反対や立場を含める際は具体的にしてください。
* 一貫した用語：常に「声明文」を使用し、「コメント」は使用しないでください。
</criteria>

<output_format format="markdown">
* **タイトルケーステーマ**：文
</output_format>`
};

/**
 * Multi-language prompt for generating overview summary (one-shot method)
 */
export const OVERVIEW_ONE_SHOT_PROMPT: Record<SupportedLanguage, string> = {
  "en": `Your job is to compose a summary of the key findings from a public discussion, based on already composed summaries corresponding to topics and subtopics identified in said discussion. These topic and subtopic summaries are based on comments and voting patterns that participants submitted as part of the discussion. You should format the results as a markdown list, to be included near the top of the final report, which shall include the complete topic and subtopic summaries. Do not pretend that you hold any of these opinions. You are not a participant in this discussion. Do not include specific numbers about how many comments were included in each topic or subtopic, as these will be included later in the final report output. You also do not need to recap the context of the conversation, as this will have already been stated earlier in the report. Where possible, prefer describing the results in terms of the "statements" submitted or the overall "conversation", rather than in terms of the participants' perspectives (Note: "comments" and "statements" are the same thing, but for the sake of this portion of the summary, only use the term "statements"). Remember: this is just one component of a larger report, and you should compose this so that it will flow naturally in the context of the rest of the report. Be clear and concise in your writing, and do not use the passive voice, or ambiguous pronouns.

The structure of the list you output should be in terms of the topic names, in the order that follows. Each list item should start in bold with topic name name (including percentage, exactly as listed below), then a colon, and then a short one or two sentence summary for the corresponding topic. The complete response should be only the markdown list, and no other text. For example, a list item might look like this:
<output_format format="markdown">* **Topic Name (45%):**  Topic summary.</output_format>
Here are the topics:
{topicNames}`,

  "zh-TW": `您的工作是根據已編寫的摘要來撰寫公開討論關鍵發現的摘要，這些摘要對應於討論中識別的主題和子主題。這些主題和子主題摘要是基於參與者作為討論一部分提交的評論和投票模式。您應該將結果格式化為 markdown 列表，包含在最終報告的頂部附近，該報告將包含完整的主題和子主題摘要。請勿假裝您持有這些意見中的任何一個。您不是此討論的參與者。請勿包含關於每個主題或子主題包含多少評論的具體數字，因為這些將在最終報告輸出中稍後包含。您也不需要重述對話的上下文，因為這將在報告的早期已經說明。在可能的情況下，請優先描述提交的「陳述」或整體「對話」的結果，而不是參與者的觀點（注意：「評論」和「陳述」是同一件事，但為了這部分摘要，只使用「陳述」一詞）。記住：這只是更大報告的一個組成部分，您應該撰寫它，使其在報告其餘部分的上下文中自然流動。在寫作中要清晰簡潔，不要使用被動語態或模糊的代詞。

您輸出的列表結構應該按照主題名稱，按照以下順序。每個列表項目應該以粗體開始，包含主題名稱（包括百分比，完全按照下面列出的），然後是冒號，然後是對應主題的簡短一兩句話摘要。完整回應應該只是 markdown 列表，沒有其他文字。例如，列表項目可能看起來像這樣：
<output_format format="markdown">* **主題名稱 (45%):** 主題摘要。</output_format>
以下是主題：
{topicNames}`,

  "zh-CN": `您的工作是根据已编写的摘要来撰写公开讨论关键发现的摘要，这些摘要对应于讨论中识别的主题和子主题。这些主题和子主题摘要是基于参与者作为讨论一部分提交的评论和投票模式。您应该将结果格式化为 markdown 列表，包含在最终报告的顶部附近，该报告将包含完整的主题和子主题摘要。请勿假装您持有这些意见中的任何一个。您不是此讨论的参与者。请勿包含关于每个主题或子主题包含多少评论的具体数字，因为这些将在最终报告输出中稍后包含。您也不需要重述对话的上下文，因为这将在报告的早期已经说明。在可能的情况下，请优先描述提交的「陈述」或整体「对话」的结果，而不是参与者的观点（注意：「评论」和「陈述」是同一件事，但为了这部分摘要，只使用「陈述」一词）。记住：这只是更大报告的一个组成部分，您应该撰写它，使其在报告其余部分的上下文中自然流動。在寫作中要清晰簡潔，不要使用被動語態或模糊的代詞。

您输出的列表结构应该按照主题名称，按照以下顺序。每个列表项目应该以粗体开始，包含主题名称（包括百分比，完全按照下面列出的），然后是冒号，然后是对应主题的简短一两句话摘要。完整回应应该只是 markdown 列表，没有其他文字。例如，列表项目可能看起来像这样：
<output_format format="markdown">* **主题名称 (45%):** 主题摘要。</output_format>
以下是主题：
{topicNames}`,

  "fr": `Votre travail consiste à composer un résumé des principales découvertes d'une discussion publique, basé sur des résumés déjà composés correspondant aux sujets et sous-sujets identifiés dans ladite discussion. Ces résumés de sujets et sous-sujets sont basés sur les commentaires et les modèles de vote que les participants ont soumis dans le cadre de la discussion. Vous devez formater les résultats sous forme de liste markdown, à inclure près du haut du rapport final, qui inclura les résumés complets des sujets et sous-sujets. Ne prétendez pas que vous détenez l'une de ces opinions. Vous n'êtes pas un participant à cette discussion. N'incluez pas de chiffres spécifiques sur le nombre de commentaires inclus dans chaque sujet ou sous-sujet, car ceux-ci seront inclus plus tard dans la sortie du rapport final. Vous n'avez pas non plus besoin de récapituler le contexte de la conversation, car cela aura déjà été énoncé plus tôt dans le rapport. Dans la mesure du possible, préférez décrire les résultats en termes de "déclarations" soumises ou de "conversation" globale, plutôt qu'en termes de perspectives des participants (Note : "commentaires" et "déclarations" sont la même chose, mais pour cette partie du résumé, utilisez uniquement le terme "déclarations"). Rappelez-vous : ce n'est qu'un composant d'un rapport plus large, et vous devez le composer pour qu'il s'intègre naturellement dans le contexte du reste du rapport. Soyez clair et concis dans votre écriture, et n'utilisez pas la voix passive ou des pronoms ambigus.

La structure de la liste que vous produisez doit être en termes de noms de sujets, dans l'ordre qui suit. Chaque élément de liste doit commencer en gras avec le nom du sujet (y compris le pourcentage, exactement comme listé ci-dessous), puis deux points, puis un résumé court d'une ou deux phrases pour le sujet correspondant. La réponse complète doit être uniquement la liste markdown, sans autre texte. Par exemple, un élément de liste pourrait ressembler à ceci :
<output_format format="markdown">* **Nom du sujet (45%) :** Résumé du sujet.</output_format>
Voici les sujets :
{topicNames}`,


  "es": `Su trabajo es componer un resumen de los hallazgos clave de una discusión pública, basado en resúmenes ya compuestos que corresponden a temas y subtemas identificados en dicha discusión. Estos resúmenes de temas y subtemas se basan en comentarios y patrones de votación que los participantes enviaron como parte de la discusión. Debe formatear los resultados como una lista markdown, para ser incluida cerca de la parte superior del informe final, que incluirá los resúmenes completos de temas y subtemas. No pretenda que sostiene alguna de estas opiniones. Usted no es un participante en esta discusión. No incluya números específicos sobre cuántos comentarios se incluyeron en cada tema o subtema, ya que estos se incluirán más tarde en la salida del informe final. Tampoco necesita recapitular el contexto de la conversación, ya que esto se habrá establecido anteriormente en el informe. Cuando sea posible, prefiera describir los resultados en términos de las "declaraciones" enviadas o la "conversación" general, en lugar de en términos de las perspectivas de los participantes (Nota: "comentarios" y "declaraciones" son lo mismo, pero por el bien de esta parte del resumen, solo use el término "declaraciones"). Recuerde: esto es solo un componente de un informe más grande, y debe componerlo para que fluya naturalmente en el contexto del resto del informe. Sea claro y conciso en su escritura, y no use la voz pasiva o pronombres ambiguos.

La estructura de la lista que produce debe estar en términos de nombres de temas, en el orden que sigue. Cada elemento de la lista debe comenzar en negrita con el nombre del tema (incluyendo el porcentaje, exactamente como se lista a continuación), luego dos puntos, luego un resumen corto de una o dos oraciones para el tema correspondiente. La respuesta completa debe ser únicamente la lista markdown, sin otro texto. Por ejemplo, un elemento de la lista podría verse así:
<output_format format="markdown">* **Nombre del Tema (45%):** Resumen del tema.</output_format>
Aquí están los temas:
{topicNames}`,


  "ja": `あなたの仕事は、既に作成された要約に基づいて公開討論の主要な発見の要約を作成することです。これらの要約は、討論で特定されたトピックとサブトピックに対応しています。これらのトピックとサブトピックの要約は、参加者が討論の一部として提出したコメントと投票パターンに基づいています。この要約はmarkdownリストとしてフォーマットされ、最終レポートの上部近くに含まれます。最終レポートには、完全なトピックとサブトピックの要約が含まれます。これらの意見のいずれかを保持しているふりをしないでください。あなたはこの討論の参加者ではありません。可能な限り、参加者の視点ではなく、提出された「声明」または全体的な「会話」の観点から結果を説明することを好んでください（注：「コメント」と「声明」は同じものですが、この要約の部分では、「声明」という用語のみを使用してください）。各トピックまたはサブトピックに含まれるコメントの数について具体的な数字を含めないでください。これらは最終レポートの出力で後ほど含まれるからです。また、会話の文脈を再説明する必要もありません。これはレポートの早い段階で既に述べられているからです。覚えておいてください：これはより大きなレポートの1つのコンポーネントにすぎず、レポートの残りの部分の文脈で自然に流れるようにこれを構成する必要があります。文章を明確で簡潔にし、受動態や曖昧な代名詞を使用しないでください。

あなたが出力するリストの構造は、以下の順序でトピック名に基づいている必要があります。各リスト項目は、トピック名（以下に正確にリストされているパーセンテージを含む）を太字で開始し、次にコロン、次に対応するトピックの短い1つまたは2つの文の要約を記述する必要があります。完全な回答は、markdownリストのみで、他のテキストは含まれない必要があります。例えば、リスト項目は次のようになります：
<output_format format="markdown">* **トピック名 (45%):** トピックの要約。</output_format>
以下がトピックです：
{topicNames}`
};

/**
 * Multi-language prompt for generating overview summary (per-topic method)
 */
export const OVERVIEW_PER_TOPIC_PROMPT: Record<SupportedLanguage, string> = {
  "en": `Your job is to compose a summary of the key findings from a public discussion, based on already composed summaries corresponding to topics and subtopics identified in said discussion. These topic and subtopic summaries are based on comments and voting patterns that participants submitted as part of the discussion. This summary will be formatted as a markdown list, to be included near the top of the final report, which shall include the complete topic and subtopic summaries. Do not pretend that you hold any of these opinions. You are not a participant in this discussion. Where possible, prefer describing the results in terms of the "statements" submitted or the overall "conversation", rather than in terms of the participants' perspectives (Note: "comments" and "statements" are the same thing, but for the sake of this portion of the summary, only use the term "statements"). Do not include specific numbers about how many comments were included in each topic or subtopic, as these will be included later in the final report output. You also do not need to recap the context of the conversation, as this will have already been stated earlier in the report. Remember: this is just one component of a larger report, and you should compose this so that it will flow naturally in the context of the rest of the report. Be clear and concise in your writing, and do not use the passive voice, or ambiguous pronouns.

Other topics will come later, but for now, your job is to compose a very short one or two sentence summary of the following topic: {topicName}. This summary will be put together into a list with other such summaries later.`,

  "zh-TW": `您的工作是根據已編寫的摘要來撰寫公開討論關鍵發現的摘要，這些摘要對應於討論中識別的主題和子主題。這些主題和子主題摘要是基於參與者作為討論一部分提交的評論和投票模式。此摘要將格式化為 markdown 列表，包含在最終報告的頂部附近，該報告將包含完整的主題和子主題摘要。請勿假裝您持有這些意見中的任何一個。您不是此討論的參與者。在可能的情況下，請優先描述提交的「陳述」或整體「對話」的結果，而不是參與者的觀點（注意：「評論」和「陳述」是同一件事，但為了這部分摘要，只使用「陳述」一詞）。請勿包含關於每個主題或子主題包含多少評論的具體數字，因為這些將在最終報告輸出中稍後包含。您也不需要重述對話的上下文，因為這將在報告的早期已經說明。記住：這只是更大報告的一個組成部分，您應該撰寫它，使其在報告其餘部分的上下文中自然流動。在寫作中要清晰簡潔，不要使用被動語態或模糊的代詞。

其他主題將稍後出現，但現在，您的工作是為以下主題撰寫一個非常簡短的一兩句話摘要：{topicName}。此摘要稍後將與其他此類摘要一起放入列表中。`,

  "zh-CN": `您的工作是根据已编写的摘要来撰写公开讨论关键发现的摘要，这些摘要对应于讨论中识别的主题和子主题。这些主题和子主题摘要是基于参与者作为讨论一部分提交的评论和投票模式。此摘要将格式化为 markdown 列表，包含在最终报告的顶部附近，该报告将包含完整的主题和子主题摘要。请勿假装您持有这些意见中的任何一个。您不是此讨论的参与者。在可能的情况下，请优先描述提交的「陈述」或整体「对话」的结果，而不是参与者的观点（注意：「评论」和「陈述」是同一件事，但为了这部分摘要，只使用「陈述」一词）。请勿包含关于每个主题或子主题包含多少评论的具体数字，因为这些将在最终报告输出中稍后包含。您也不需要重述对话的上下文，因为这将在报告的早期已经说明。记住：这只是更大报告的一个组成部分，您应该撰写它，使其在报告其余部分的上下文中自然流動。在寫作中要清晰簡潔，不要使用被動語態或模糊的代詞。

其他主题将稍后出现，但现在，您的工作是为以下主题撰写一个非常简短的一两句话摘要：{topicName}。此摘要稍后将与其他此类摘要一起放入列表中。`,

  "fr": `Votre travail consiste à composer un résumé des principales découvertes d'une discussion publique, basé sur des résumés déjà composés correspondant aux sujets et sous-sujets identifiés dans ladite discussion. Ces résumés de sujets et sous-sujets sont basés sur les commentaires et les modèles de vote que les participants ont soumis dans le cadre de la discussion. Ce résumé sera formaté sous forme de liste markdown, à inclure près du haut du rapport final, qui inclura les résumés complets des sujets et sous-sujets. Ne prétendez pas que vous détenez l'une de ces opinions. Vous n'êtes pas un participant à cette discussion. Dans la mesure du possible, préférez décrire les résultats en termes de "déclarations" soumises ou de "conversation" globale, plutôt qu'en termes de perspectives des participants (Note : "commentaires" et "déclarations" sont la même chose, mais pour cette partie du résumé, utilisez uniquement le terme "déclarations"). N'incluez pas de chiffres spécifiques sur le nombre de commentaires inclus dans chaque sujet ou sous-sujet, car ceux-ci seront inclus plus tard dans la sortie du rapport final. Vous n'avez pas non plus besoin de récapituler le contexte de la conversation, car cela aura déjà été énoncé plus tôt dans le rapport. Rappelez-vous : ce n'est qu'un composant d'un rapport plus large, et vous devez le composer pour qu'il s'intègre naturellement dans le contexte du reste du rapport. Soyez clair et concis dans votre écriture, et n'utilisez pas la voix passive ou des pronoms ambigus.

D'autres sujets viendront plus tard, mais pour l'instant, votre travail consiste à composer un résumé très court d'une ou deux phrases du sujet suivant : {topicName}. Ce résumé sera plus tard assemblé dans une liste avec d'autres résumés de ce type.`,

  "es": `Su trabajo es componer un resumen de los hallazgos clave de una discusión pública, basado en resúmenes ya compuestos que corresponden a temas y subtemas identificados en dicha discusión. Estos resúmenes de temas y subtemas se basan en comentarios y patrones de votación que los participantes enviaron como parte de la discusión. Este resumen se formateará como una lista markdown, para ser incluida cerca de la parte superior del informe final, que incluirá los resúmenes completos de temas y subtemas. No pretenda que sostiene alguna de estas opiniones. Usted no es un participante en esta discusión. Cuando sea posible, prefiera describir los resultados en términos de las "declaraciones" enviadas o la "conversación" general, en lugar de en términos de las perspectivas de los participantes (Nota: "comentarios" y "declaraciones" son lo mismo, pero por el bien de esta parte del resumen, solo use el término "declaraciones"). No incluya números específicos sobre cuántos comentarios se incluyeron en cada tema o subtema, ya que estos se incluirán más tarde en la salida del informe final. Tampoco necesita recapitular el contexto de la conversación, ya que esto se habrá establecido anteriormente en el informe. Recuerde: esto es solo un componente de un informe más grande, y debe componerlo para que fluya naturalmente en el contexto del resto del informe. Sea claro y conciso en su escritura, y no use la voz pasiva o pronombres ambiguos.

Otros temas vendrán más tarde, pero por ahora, su trabajo es componer un resumen muy corto de una o dos oraciones del siguiente tema: {topicName}. Este resumen se pondrá más tarde en una lista con otros resúmenes de este tipo.`,

  "ja": `あなたの仕事は、既に作成された要約に基づいて公開討論の主要な発見の要約を作成することです。これらの要約は、討論で特定されたトピックとサブトピックに対応しています。これらのトピックとサブトピックの要約は、参加者が討論の一部として提出したコメントと投票パターンに基づいています。この要約はmarkdownリストとしてフォーマットされ、最終レポートの上部近くに含まれます。最終レポートには、完全なトピックとサブトピックの要約が含まれます。これらの意見のいずれかを保持しているふりをしないでください。あなたはこの討論の参加者ではありません。可能な限り、参加者の視点ではなく、提出された「声明」または全体的な「会話」の観点から結果を説明することを好んでください（注：「コメント」と「声明」は同じものですが、この要約の部分では、「声明」という用語のみを使用してください）。各トピックまたはサブトピックに含まれるコメントの数について具体的な数字を含めないでください。これらは最終レポートの出力で後ほど含まれるからです。また、会話の文脈を再説明する必要もありません。これはレポートの早い段階で既に述べられているからです。覚えておいてください：これはより大きなレポートの1つのコンポーネントにすぎず、レポートの残りの部分の文脈で自然に流れるようにこれを構成する必要があります。受動態を使用しないでください。曖昧な代名詞を使用しないでください。明確にしてください。箇条書きや特別なフォーマットを生成しないでください。無駄話をしないでください。

他のトピックは後で来ますが、今のところ、あなたの仕事は以下のトピックについて非常に短い1つまたは2つの文の要約を作成することです：{topicName}。この要約は後で他の同様の要約と一緒にリストにまとめられます。`
};

/**
 * Multi-language prompt for learning topics from comments
 */
export const LEARN_TOPICS_PROMPT: Record<SupportedLanguage, string> = {
  "en": `Analyze the following comments and identify common topics.
Consider the granularity of topics: too few topics may oversimplify the content and miss important nuances, while too many topics may lead to redundancy and make the overall structure less clear.
Aim for a balanced number of topics that effectively summarizes the key themes without excessive detail.
After analysis of the comments, determine the optimal number of topics to represent the content effectively.
Justify why having fewer topics would be less optimal (potentially oversimplifying and missing key nuances), and why having more topics would also be less optimal (potentially leading to redundancy and a less clear overall structure).
After determining the optimal number of topics, identify those topics.

IMPORTANT: 
- Do NOT create a topic named "Other" or "Miscellaneous" or similar catch-all names.
- Each topic should have a specific, descriptive name that clearly represents the content.
- Output only the actual topics found in the comments, with clear, meaningful names.
- Use the exact JSON schema format specified: [{"name": "Topic Name"}]`,

  "zh-TW": `分析以下評論並識別共同主題。
考慮主題的粒度：主題太少可能會過度簡化內容並錯過重要細微差別，而主題太多可能會導致冗餘並使整體結構不太清晰。
目標是平衡數量的主題，能夠有效總結關鍵主題而不過度詳細。
分析評論後，確定最佳主題數量以有效表示內容。
證明為什麼主題較少會不太理想（可能過度簡化並錯過關鍵細微差別），以及為什麼主題較多也會不太理想（可能導致冗餘和不太清晰的整體結構）。
確定最佳主題數量後，識別這些主題。

重要事項：
- 請勿創建名為「其他」或「雜項」或類似包羅萬象名稱的主題。
- 每個主題都應該有一個具體、描述性的名稱，清楚代表內容。
- 僅輸出在評論中找到的實際主題，具有清晰、有意義的名稱。
- 使用指定的確切 JSON 架構格式：[{"name": "主題名稱"}]`,

  "zh-CN": `分析以下评论并识别共同主题。
考虑主题的粒度：主题太少可能会过度简化内容并错过重要细微差别，而主题太多可能会导致冗余并使整体结构不太清晰。
目标是平衡数量的主题，能够有效总结关键主题而不过度详细。
分析评论后，确定最佳主题数量以有效表示内容。
证明为什么主题较少会不太理想（可能过度简化并错过关键细微差别），以及为什么主题较多也会不太理想（可能导致冗余和不太清晰的整体结构）。
确定最佳主题数量后，识别这些主题。

重要事项：
- 请勿创建名为「其他」或「杂项」或类似包罗万象名称的主题。
- 每个主题都应该有一个具体、描述性的名称，清楚代表内容。
- 仅输出在评论中找到的实际主题，具有清晰、有意义的名称。
- 使用指定的确切 JSON 架构格式：[{"name": "主题名称"}]`,

  "fr": `Analysez les commentaires suivants et identifiez les sujets communs.
Considérez la granularité des sujets : trop peu de sujets peuvent simplifier à l'excès le contenu et manquer des nuances importantes, tandis que trop de sujets peuvent mener à la redondance et rendre la structure globale moins claire.
Visez un nombre équilibré de sujets qui résume efficacement les thèmes clés sans détails excessifs.
Après analyse des commentaires, déterminez le nombre optimal de sujets pour représenter efficacement le contenu.
Justifiez pourquoi avoir moins de sujets serait moins optimal (potentiellement simplifier à l'excès et manquer des nuances clés), et pourquoi avoir plus de sujets serait également moins optimal (potentiellement mener à la redondance et une structure globale moins claire).
Après avoir déterminé le nombre optimal de sujets, identifiez ces sujets.

IMPORTANT :
- Ne créez PAS de sujet nommé "Autre" ou "Divers" ou des noms similaires fourre-tout.
- Chaque sujet doit avoir un nom spécifique et descriptif qui représente clairement le contenu.
- N'outputez que les sujets réels trouvés dans les commentaires, avec des noms clairs et significatifs.
- Utilisez le format de schéma JSON exact spécifié : [{"name": "Nom du Sujet"}]`,

  "es": `Analice los siguientes comentarios e identifique temas comunes.
Considere la granularidad de los temas: muy pocos temas pueden simplificar en exceso el contenido y perder matices importantes, mientras que demasiados temas pueden llevar a la redundancia y hacer que la estructura general sea menos clara.
Aim for a balanced number of topics that effectively summarizes the key themes without excessive detail.
Después del análisis de los comentarios, determine el número óptimo de temas para representar efectivamente el contenido.
Justifique por qué tener menos temas sería menos óptimo (potencialmente simplificando en exceso y perdiendo matices clave), y por qué tener más temas también sería menos óptimo (potencialmente llevando a la redundancia y una estructura general menos clara).
Después de determinar el número óptimo de temas, identifique esos temas.

IMPORTANTE:
- NO cree un tema llamado "Otros" o "Misceláneos" o nombres similares de captura general.
- Cada tema debe tener un nombre específico y descriptivo que represente claramente el contenido.
- Solo outpute los temas reales encontrados en los comentarios, con nombres claros y significativos.
- Use el formato de esquema JSON exacto especificado: [{"name": "Nombre del Tema"}]`,

  "ja": `以下のコメントを分析し、共通のトピックを特定してください。
トピックの粒度を考慮してください：トピックが少なすぎると内容を過度に簡素化し、重要なニュアンスを見逃す可能性があります。一方、トピックが多すぎると冗長性を招き、全体的な構造が不明確になる可能性があります。
内容を効果的に要約するバランスの取れた数のトピックを目指し、過度な詳細は避けてください。
コメントの分析後、内容を効果的に表現する最適なトピック数を決定してください。
トピックが少なすぎることが最適でない理由（内容を過度に簡素化し、重要なニュアンスを見逃す可能性）と、トピックが多すぎることも最適でない理由（冗長性を招き、全体的な構造が不明確になる可能性）を証明してください。
最適なトピック数を決定した後、それらのトピックを特定してください。

重要：
- 「その他」や「雑多」、または類似の包括的な名前のトピックを作成しないでください。
- 各トピックは、内容を明確に表現する具体的で説明的な名前を持つ必要があります。
- コメントで見つかった実際のトピックのみを出力し、明確で意味のある名前を使用してください。
- 指定された正確なJSONスキーマ形式を使用してください：[{"name": "トピック名"}]`
};

/**
 * Multi-language prompt for learning subtopics from comments
 */
export const LEARN_SUBTOPICS_PROMPT: Record<SupportedLanguage, string> = {
  "en": `Analyze the following comments and identify common subtopics within the following overarching topic: "{parentTopicName}".
Consider the granularity of subtopics: too few subtopics may oversimplify the content and miss important nuances, while too many subtopics may lead to redundancy and make the overall structure less clear.
Aim for a balanced number of subtopics that effectively summarizes the key themes without excessive detail.
After analysis of the comments, determine the optimal number of subtopics to represent the content effectively.
Justify why having fewer subtopics would be less optimal (potentially oversimplifying and missing key nuances), and why having more subtopics would also be less optimal (potentially leading to redundancy and a less clear overall structure).
After determining the optimal number of subtopics, identify those subtopics.

CRITICAL REQUIREMENTS:
- You MUST return at least one subtopic. NEVER return an empty array or empty content.
- If you cannot identify distinct subtopics, create at least one subtopic that captures the main theme of the comments.
- No subtopics should have the same name as the overarching topic.
- There are other overarching topics that are being used on different sets of comments, do not use these overarching topic names as identified subtopics names: {otherTopicNames}

Example of Incorrect Output:

[
  {
    "name": "Economic Development",
    "subtopics": [
        { "name": "Job Creation" },
        { "name": "Business Growth" },
        { "name": "Small Business Development" },
        { "name": "Small Business Marketing" } // Incorrect: Too closely related to the "Small Business Development" subtopic
        { "name": "Infrastructure & Transportation" } // Incorrect: This is the name of a main topic
      ]
  }
]`,

  "zh-TW": `分析以下評論並識別以下總體主題內的共同子主題：「{parentTopicName}」。
考慮子主題的粒度：子主題太少可能會過度簡化內容並錯過重要細微差別，而子主題太多可能會導致冗餘並使整體結構不太清晰。
目標是平衡數量的子主題，能夠有效總結關鍵主題而不過度詳細。
分析評論後，確定最佳子主題數量以有效表示內容。
證明為什麼子主題較少會不太理想（可能過度簡化並錯過關鍵細微差別），以及為什麼子主題較多也會不太理想（可能導致冗餘和不太清晰的整體結構）。
確定最佳子主題數量後，識別這些子主題。

關鍵要求：
- 您必須返回至少一個子主題。絕對不要返回空陣列或空白內容。
- 如果您無法識別不同的子主題，請創建至少一個能捕捉評論主要主題的子主題。
- 任何子主題都不應該與總體主題同名。
- 有其他總體主題正在不同的評論集合中使用，請勿將這些總體主題名稱用作識別的子主題名稱：{otherTopicNames}

錯誤輸出示例：

[
  {
    "name": "經濟發展",
    "subtopics": [
        { "name": "創造就業" },
        { "name": "業務增長" },
        { "name": "小企業發展" },
        { "name": "小企業營銷" } // 錯誤：與「小企業發展」子主題過於密切相關
        { "name": "基礎設施與交通" } // 錯誤：這是主要主題的名稱
      ]
  }
]`,

  "zh-CN": `分析以下评论并识别以下总体主题内的共同子主题：「{parentTopicName}」。
考虑子主题的粒度：子主题太少可能会过度简化内容并错过重要细微差别，而子主题太多可能会导致冗余并使整体结构不太清晰。
目标是平衡数量的子主题，能够有效总结关键主题而不过度详细。
分析评论后，确定最佳子主题数量以有效表示内容。
证明为什么子主题较少会不太理想（可能过度简化并错过关键细微差别），以及为什么子主题较多也会不太理想（可能导致冗余和不太清晰的整体结构）。
确定最佳子主题数量后，识别这些子主题。

关键要求：
- 您必须返回至少一个子主题。绝对不要返回空数组或空白内容。
- 如果您无法识别不同的子主题，请创建至少一个能捕捉评论主要主题的子主题。
- 任何子主题都不应该与总体主题同名。
- 有其他总体主题正在不同的评论集合中使用，请勿将这些总体主题名称用作识别的子主题名称：{otherTopicNames}

错误输出示例：

[
  {
    "name": "经济发展",
    "subtopics": [
        { "name": "创造就业" },
        { "name": "业务增长" },
        { "name": "小企业发展" },
        { "name": "小企业营销" } // 错误：与「小企业发展」子主题过于密切相关
        { "name": "基础设施与交通" } // 错误：这是主要主题的名称
      ]
  }
]`,

  "fr": `Analysez les commentaires suivants et identifiez les sous-sujets communs dans le sujet général suivant : "{parentTopicName}".
Considérez la granularité des sous-sujets : trop peu de sous-sujets peuvent simplifier à l'excès le contenu et manquer des nuances importantes, tandis que trop de sous-sujets peuvent mener à la redondance et rendre la structure globale moins claire.
Visez un nombre équilibré de sous-sujets qui résume efficacement les thèmes clés sans détails excessifs.
Après analyse des commentaires, déterminez le nombre optimal de sous-sujets pour représenter efficacement le contenu.
Justifiez pourquoi avoir moins de sous-sujets serait moins optimal (potentiellement simplifier à l'excès et manquer des nuances clés), et pourquoi avoir plus de sous-sujets serait également moins optimal (potentiellement mener à la redondance et une structure globale moins claire).
Après avoir déterminé le nombre optimal de sous-sujets, identifiez ces sous-sujets.

Exigences critiques :
- Vous DEVEZ retourner au moins un sous-sujet. NE retournez JAMAIS un tableau vide ou un contenu vide.
- Si vous ne pouvez pas identifier des sous-sujets distincts, créez au moins un sous-sujet qui capture le thème principal des commentaires.
- Aucun sous-sujet ne doit avoir le même nom que le sujet général.
- Il y a d'autres sujets généraux qui sont utilisés sur différents ensembles de commentaires, n'utilisez pas ces noms de sujets généraux comme noms de sous-sujets identifiés : {otherTopicNames}

Exemple de sortie incorrecte :

[
  {
    "name": "Développement économique",
    "subtopics": [
        { "name": "Création d'emplois" },
        { "name": "Croissance des entreprises" },
        { "name": "Développement des petites entreprises" },
        { "name": "Marketing des petites entreprises" } // Incorrect : Trop étroitement lié au sous-sujet "Développement des petites entreprises"
        { "name": "Infrastructure et transport" } // Incorrect : C'est le nom d'un sujet principal
      ]
  }
]`,

  "es": `Analice los siguientes comentarios e identifique subtemas comunes dentro del siguiente tema general: "{parentTopicName}".
Considere la granularidad de los subtemas: muy pocos subtemas pueden simplificar en exceso el contenido y perder matices importantes, mientras que demasiados subtemas pueden llevar a la redundancia y hacer que la estructura general sea menos clara.
Aim for a balanced number of subtopics that effectively summarizes the key themes without excessive detail.
Después del análisis de los comentarios, determine el número óptimo de subtemas para representar efectivamente el contenido.
Justifique por qué tener menos subtemas sería menos óptimo (potencialmente simplificando en exceso y perdiendo matices clave), y por qué tener más subtemas también sería menos óptimo (potencialmente llevando a la redundancia y una estructura general menos clara).
Después de determinar el número óptimo de subtemas, identifique esos subtemas.

Requisitos críticos:
- DEBE devolver al menos un subtema. NUNCA devuelva un array vacío o contenido vacío.
- Si no puede identificar subtemas distintos, cree al menos un subtema que capture el tema principal de los comentarios.
- Ningún subtema debe tener el mismo nombre que el tema general.
- Hay otros temas generales que se están utilizando en diferentes conjuntos de comentarios, no use estos nombres de temas generales como nombres de subtemas identificados: {otherTopicNames}

Ejemplo de salida incorrecta:

[
  {
    "name": "Desarrollo Económico",
    "subtopics": [
        { "name": "Creación de Empleos" },
        { "name": "Crecimiento Empresarial" },
        { "name": "Desarrollo de Pequeñas Empresas" },
        { "name": "Marketing de Pequeñas Empresas" } // Incorrecto: Demasiado estrechamente relacionado con el subtema "Desarrollo de Pequeñas Empresas"
        { "name": "Infraestructura y Transporte" } // Incorrecto: Este es el nombre de un tema principal
      ]
  }
]`,

  "ja": `以下のコメントを分析し、以下の包括的なトピック内の共通のサブトピックを特定してください：「{parentTopicName}」。
サブトピックの粒度を考慮してください：サブトピックが少なすぎると内容を過度に簡素化し、重要なニュアンスを見逃す可能性があります。一方、サブトピックが多すぎると冗長性を招き、全体的な構造が不明確になる可能性があります。
内容を効果的に要約するバランスの取れた数のサブトピックを目指し、過度な詳細は避けてください。
コメントの分析後、内容を効果的に表現する最適なサブトピック数を決定してください。
サブトピックが少なすぎることが最適でない理由（内容を過度に簡素化し、重要なニュアンスを見逃す可能性）と、サブトピックが多すぎることも最適でない理由（冗長性を招き、全体的な構造が不明確になる可能性）を証明してください。
最適なサブトピック数を決定した後、それらのサブトピックを特定してください。

重要な要件：
- 少なくとも1つのサブトピックを返す必要があります。空の配列や空のコンテンツを返してはいけません。
- 異なるサブトピックを識別できない場合は、コメントの主要なテーマを捉える少なくとも1つのサブトピックを作成してください。
- サブトピックは包括的なトピックと同じ名前を持つべきではありません。
- 異なるコメントセットで使用されている他の包括的なトピックがあります。これら包括的なトピック名を識別されたサブトピック名として使用しないでください：{otherTopicNames}

誤った出力の例：

[
  {
    "name": "経済発展",
    "subtopics": [
        { "name": "雇用創出" },
        { "name": "事業成長" },
        { "name": "小企業発展" },
        { "name": "小企業マーケティング" } // 誤り：「小企業発展」サブトピックと密接に関連しすぎている
        { "name": "インフラと交通" } // 誤り：これはメイントピックの名前です
      ]
  }
]`
};

/**
 * Get the localized prompt for learning topics
 * @param language The target language
 * @returns The localized prompt for learning topics
 */
export function getLearnTopicsPrompt(language: SupportedLanguage): string {
  console.log(`[DEBUG] getLearnTopicsPrompt() language: ${language}`);
  return LEARN_TOPICS_PROMPT[language] || LEARN_TOPICS_PROMPT["en"];
}

/**
 * Get the localized prompt for learning subtopics
 * @param language The target language
 * @param parentTopicName The name of the parent topic
 * @param otherTopicNames The names of other topics to avoid
 * @returns The localized prompt for learning subtopics
 */
export function getLearnSubtopicsPrompt(
  language: SupportedLanguage, 
  parentTopicName: string, 
  otherTopicNames: string
): string {
  console.log(`[DEBUG] getLearnSubtopicsPrompt() language: ${language}`);
  const prompt = LEARN_SUBTOPICS_PROMPT[language] || LEARN_SUBTOPICS_PROMPT["en"];
  return prompt
    .replace("{parentTopicName}", parentTopicName)
    .replace("{otherTopicNames}", otherTopicNames);
}

/**
 * Get the localized prompt for themes generation
 * @param language The target language
 * @param topicName The name of the topic to replace in the prompt
 * @returns The localized prompt with topic name replaced
 */
export function getThemesPrompt(language: SupportedLanguage, topicName: string): string {
  console.log(`[DEBUG] getThemesPrompt() language: ${language}`);
  const prompt = THEMES_PROMPT[language] || THEMES_PROMPT["en"];
  return prompt.replace("{topicName}", topicName);
}

/**
 * Get the localized prompt for overview summary (one-shot method)
 * @param language The target language
 * @param topicNames The list of topic names to replace in the prompt
 * @returns The localized prompt with topic names replaced
 */
export function getOverviewOneShotPrompt(language: SupportedLanguage, topicNames: string[]): string {
  console.log(`[DEBUG] getOverviewOneShotPrompt() language: ${language}`);
  const prompt = OVERVIEW_ONE_SHOT_PROMPT[language] || OVERVIEW_ONE_SHOT_PROMPT["en"];
  return prompt.replace("{topicNames}", topicNames.map((s) => "* " + s).join("\n"));
}

/**
 * Get the localized prompt for overview summary (per-topic method)
 * @param language The target language
 * @param topicName The name of the topic to replace in the prompt
 * @returns The localized prompt with topic name replaced
 */
export function getOverviewPerTopicPrompt(language: SupportedLanguage, topicName: string): string {
  console.log(`[DEBUG] getOverviewPerTopicPrompt() language: ${language}`);
  const prompt = OVERVIEW_PER_TOPIC_PROMPT[language] || OVERVIEW_PER_TOPIC_PROMPT["en"];
  return prompt.replace("{topicName}", topicName);
}

/**
 * Multi-language prompt for differences of opinion instructions
 */
export const DIFFERENCES_OF_OPINION_INSTRUCTIONS: Record<SupportedLanguage, string> = {
  "en": `You are going to be presented with several comments from a discussion on which there were differing opinions, ` +
  `as well as a summary of points of common ground from this discussion. Your job is summarize the ideas ` +
  `contained in the comments, keeping in mind the points of common ground as backgrounnd in describing ` +
  `the differences of opinion. Do not pretend that you hold any of these opinions. You are not a ` +
  `participant in this discussion. Write a concise summary of these comments that is at least ` +
  `one sentence and at most five sentences long. Refer to the people who made these comments as ` +
  `participants, not commenters.  Do not talk about how strongly they disagree with these ` +
  `comments. Use complete sentences. Do not use the passive voice. Do not use ambiguous pronouns. Be clear. ` +
  `Do not generate bullet points or special formatting. Do not yap.

Do not assume that these comments were written by different participants. These comments could be from ` +
  `the same participant, so do not say some participants prosed one things while other ` +
  `participants proposed another.  Do not say "Some participants proposed X while others Y".  ` +
  `Instead say "One statement proposed X while another Y"

Where the difference of opinion comments refer to topics that are also covered in the common ground ` +
  `summary, your output should begin in some variant of the form "While there was broad support for ..., ` +
  `opinions differed with respect to ...". When this is not the case, you can beging simple as ` +
  `"There was disagreement ..." or something similar to contextualize that the comments you are ` +
  `summarizing had mixed support.`,

  "zh-TW": `您將被呈現來自討論中意見分歧的幾個評論，以及該討論中共同點的摘要。您的工作是總結這些評論中包含的想法，在描述意見分歧時要記住共同點作為背景。請勿假裝您持有這些意見中的任何一個。您不是此討論的參與者。撰寫這些評論的簡潔摘要，至少一個句子，最多五個句子。將發表這些評論的人稱為參與者，而不是評論者。請勿談論他們對這些評論的不同意程度。使用完整句子。不要使用被動語態。不要使用模糊的代詞。要清晰。不要生成項目符號或特殊格式。不要廢話。

請勿假設這些評論是由不同參與者撰寫的。這些評論可能來自同一個參與者，所以請勿說一些參與者提出一件事，而其他參與者提出另一件事。請勿說「一些參與者提出X，而其他人提出Y」。相反，請說「一個聲明提出X，而另一個提出Y」。

當意見分歧的評論涉及在共同點摘要中也涵蓋的主題時，您的輸出應該以「雖然對...有廣泛支持，但對...的意見有所不同」的形式開始。當不是這種情況時，您可以簡單地開始為「存在分歧...」或類似的內容，以說明您正在總結的評論有混合支持。`,

  "zh-CN": `您将被呈现来自讨论中意见分歧的几个评论，以及该讨论中共同点的摘要。您的工作是总结这些评论中包含的想法，在描述意见分歧时要记住共同点作为背景。请勿假装您持有这些意见中的任何一个。您不是此讨论的参与者。撰写这些评论的简洁摘要，至少一个句子，最多五个句子。将发表这些评论的人称为参与者，而不是评论者。请勿谈论他们对这些评论的不同意程度。使用完整句子。不要使用被动语态。不要使用模糊的代词。要清晰。不要生成项目符号或特殊格式。不要废话。

请勿假设这些评论是由不同参与者撰写的。这些评论可能来自同一个参与者，所以请勿说一些参与者提出一件事，而其他参与者提出另一件事。请勿说「一些参与者提出X，而其他人提出Y」。相反，请说「一个声明提出X，而另一个提出Y」。

当意见分歧的评论涉及在共同点摘要中也涵盖的主题时，您的输出应该以「虽然对...有广泛支持，但对...的意见有所不同」的形式开始。当不是这种情况时，您可以简单地开始为「存在分歧...」或类似的内容，以说明您正在总结的评论有混合支持。`,

  "fr": `Votre travail consiste à composer un résumé des principales découvertes d'une discussion publique, basé sur des résumés déjà composés correspondant aux sujets et sous-sujets identifiés dans ladite discussion. Ces résumés de sujets et sous-sujets sont basés sur les commentaires et les modèles de vote que les participants ont soumis dans le cadre de la discussion. Vous devez formater les résultats sous forme de liste markdown, à inclure près du haut du rapport final, qui inclura les résumés complets des sujets et sous-sujets. Ne prétendez pas que vous détenez l'une de ces opinions. Vous n'êtes pas un participant à cette discussion. N'incluez pas de chiffres spécifiques sur le nombre de commentaires inclus dans chaque sujet ou sous-sujet, car ceux-ci seront inclus plus tard dans la sortie du rapport final. Vous n'avez pas non plus besoin de récapituler le contexte de la conversation, car cela aura déjà été énoncé plus tôt dans le rapport. Dans la mesure du possible, préférez décrire les résultats en termes de "déclarations" soumises ou de "conversation" globale, plutôt qu'en termes de perspectives des participants (Note : "commentaires" et "déclarations" sont la même chose, mais pour cette partie du résumé, utilisez uniquement le terme "déclarations"). Rappelez-vous : ce n'est qu'un composant d'un rapport plus large, et vous devez le composer pour qu'il s'intègre naturellement dans le contexte du reste du rapport. Soyez clair et concis dans votre écriture, et n'utilisez pas la voix passive ou des pronoms ambigus.

La structure de la liste que vous produisez doit être en termes de noms de sujets, dans l'ordre qui suit. Chaque élément de liste doit commencer en gras avec le nom du sujet (y compris le pourcentage, exactement comme listé ci-dessous), puis deux points, puis un résumé court d'une ou deux phrases pour le sujet correspondant. La réponse complète doit être uniquement la liste markdown, sans autre texte. Par exemple, un élément de liste pourrait ressembler à ceci :
<output_format format="markdown">* **Nom du sujet (45%) :** Résumé du sujet.</output_format>
Voici les sujets :
{topicNames}`,


  "es": `Su trabajo es componer un resumen de los hallazgos clave de una discusión pública, basado en resúmenes ya compuestos que corresponden a temas y subtemas identificados en dicha discusión. Estos resúmenes de temas y subtemas se basan en comentarios y patrones de votación que los participantes enviaron como parte de la discusión. Debe formatear los resultados como una lista markdown, para ser incluida cerca de la parte superior del informe final, que incluirá los resúmenes completos de temas y subtemas. No pretenda que sostiene alguna de estas opiniones. Usted no es un participante en esta discusión. No incluya números específicos sobre cuántos comentarios se incluyeron en cada tema o subtema, ya que estos se incluirán más tarde en la salida del informe final. Tampoco necesita recapitular el contexto de la conversación, ya que esto se habrá establecido anteriormente en el informe. Cuando sea posible, prefiera describir los resultados en términos de las "declaraciones" enviadas o la "conversación" general, en lugar de en términos de las perspectivas de los participantes (Nota: "comentarios" y "declaraciones" son lo mismo, pero por el bien de esta parte del resumen, solo use el término "declaraciones"). Recuerde: esto es solo un componente de un informe más grande, y debe componerlo para que fluya naturalmente en el contexto del resto del informe. Sea claro y conciso en su escritura, y no use la voz pasiva o pronombres ambiguos.

La estructura de la lista que produce debe estar en términos de nombres de temas, en el orden que sigue. Cada elemento de la lista debe comenzar en negrita con el nombre del tema (incluyendo el porcentaje, exactamente como se lista a continuación), luego dos puntos, luego un resumen corto de una o dos oraciones para el tema correspondiente. La respuesta completa debe ser únicamente la lista markdown, sin otro texto. Por ejemplo, un elemento de la lista podría verse así:
<output_format format="markdown">* **Nombre del Tema (45%):** Resumen del tema.</output_format>
Aquí están los temas:
{topicNames}`,


  "ja": `あなたの仕事は、既に作成された要約に基づいて公開討論の主要な発見の要約を作成することです。これらの要約は、討論で特定されたトピックとサブトピックに対応しています。これらのトピックとサブトピックの要約は、参加者が討論の一部として提出したコメントと投票パターンに基づいています。この要約はmarkdownリストとしてフォーマットされ、最終レポートの上部近くに含まれます。最終レポートには、完全なトピックとサブトピックの要約が含まれます。これらの意見のいずれかを保持しているふりをしないでください。あなたはこの討論の参加者ではありません。可能な限り、参加者の視点ではなく、提出された「声明」または全体的な「会話」の観点から結果を説明することを好んでください（注：「コメント」と「声明」は同じものですが、この要約の部分では、「声明」という用語のみを使用してください）。各トピックまたはサブトピックに含まれるコメントの数について具体的な数字を含めないでください。これらは最終レポートの出力で後ほど含まれるからです。また、会話の文脈を再説明する必要もありません。これはレポートの早い段階で既に述べられているからです。覚えておいてください：これはより大きなレポートの1つのコンポーネントにすぎず、レポートの残りの部分の文脈で自然に流れるようにこれを構成する必要があります。文章を明確で簡潔にし、受動態や曖昧な代名詞を使用しないでください。

あなたが出力するリストの構造は、以下の順序でトピック名に基づいている必要があります。各リスト項目は、トピック名（以下に正確にリストされているパーセンテージを含む）を太字で開始し、次にコロン、次に対応するトピックの短い1つまたは2つの文の要約を記述する必要があります。完全な回答は、markdownリストのみで、他のテキストは含まれない必要があります。例えば、リスト項目は次のようになります：
<output_format format="markdown">* **トピック名 (45%):** トピックの要約。</output_format>
以下がトピックです：
{topicNames}`
};

/**
 * Multi-language prompt for differences of opinion single comment instructions
 */
export const DIFFERENCES_OF_OPINION_SINGLE_COMMENT_INSTRUCTIONS: Record<SupportedLanguage, string> = {
  "en": `You are going to be presented with a single comment from a discussion on which there were differing opinions, ` +
  `as well as a summary of points of common ground from this discussion. ` +
  `Your job is to rewrite this comment to summarize the main points or ideas it is trying to make, clearly and without embellishment,` +
  `keeping in mind the points of common ground as backgrounnd in describing the differences of opinion participants had in relation to this comment. ` +
  `Do not pretend that you hold opinions. You are not a participant in this discussion. ` +
  `Write your summary as a single complete sentence.` +
  `Refer to the people who made these comments as participants, not commenters. ` +
  `Do not talk about how strongly they disagree with these comments. Do not use the passive voice. Do not use ambiguous pronouns. Be clear. ` +
  `Do not generate bullet points or special formatting. Do not yap.

Where the difference of opinion comments refer to topics that are also covered in the common ground ` +
  `summary, your output should begin in some variant of the form "While there was broad support for ..., ` +
  `opinions differed with respect to ...". When this is not the case, you can beging simple as ` +
  `"There was disagreement ..." or something similar to contextualize that the comments you are ` +
  `summarizing had mixed support.`,

  "zh-TW": `您將被呈現來自討論中意見分歧的單個評論，以及該討論中共同點的摘要。您的工作是重寫此評論，以總結它試圖表達的主要觀點或想法，清晰且不加修飾，在描述參與者對此評論的意見分歧時要記住共同點作為背景。請勿假裝您持有意見。您不是此討論的參與者。將您的摘要寫成單個完整句子。將發表這些評論的人稱為參與者，而不是評論者。請勿談論他們對這些評論的不同意程度。不要使用被動語態。不要使用模糊的代詞。要清晰。不要生成項目符號或特殊格式。不要廢話。

當意見分歧的評論涉及在共同點摘要中也涵蓋的主題時，您的輸出應該以「雖然對...有廣泛支持，但對...的意見有所不同」的形式開始。當不是這種情況時，您可以簡單地開始為「存在分歧...」或類似的內容，以說明您正在總結的評論有混合支持。`,

  "zh-CN": `您将被呈现来自讨论中意见分歧的单个评论，以及该讨论中共同点的摘要。您的工作是重写此评论，以总结它试图表达的主要观点或想法，清晰且不加修饰，在描述参与者对此评论的意见分歧时要记住共同点作为背景。请勿假装您持有意见。您不是此讨论的参与者。将您的摘要写成单个完整句子。将发表这些评论的人称为参与者，而不是评论者。请勿谈论他们对这些评论的不同意程度。不要使用被动语态。不要使用模糊的代词。要清晰。不要生成项目符号或特殊格式。不要废话。

当意见分歧的评论涉及在共同点摘要中也涵盖的主题时，您的输出应该以「虽然对...有广泛支持，但对...的意见有所不同」的形式开始。当不是这种情况时，您可以简单地开始为「存在分歧...」或类似的内容，以说明您正在总结的评论有混合支持。`,

  "fr": `Vous allez être présenté avec un seul commentaire d'une discussion sur laquelle il y avait des opinions divergentes, ` +
  `ainsi qu'un résumé des points de terrain d'entente de cette discussion. ` +
  `Votre travail est de réécrire ce commentaire pour résumer les points principaux ou les idées qu'il essaie de faire valoir, clairement et sans embellissement,` +
  `en gardant à l'esprit les points de terrain d'entente comme arrière-plan en décrivant les différences d'opinion que les participants avaient par rapport à ce commentaire. ` +
  `Ne prétendez pas que vous détenez des opinions. Vous n'êtes pas un participant à cette discussion. ` +
  `Rédigez votre résumé comme une seule phrase complète.` +
  `Référez-vous aux personnes qui ont fait ces commentaires comme participants, pas comme commentateurs. ` +
  `Ne parlez pas de la force avec laquelle ils sont en désaccord avec ces commentaires. N'utilisez pas la voix passive. N'utilisez pas de pronoms ambigus. Soyez clair. ` +
  `Ne générez pas de puces ou de formatage spécial. Ne bavardez pas.

Lorsque les commentaires de différence d'opinion se réfèrent à des sujets qui sont également couverts dans le résumé du terrain d'entente, ` +
  `votre sortie devrait commencer par une variante de la forme "Bien qu'il y ait eu un large soutien pour ..., ` +
  `les opinions différaient en ce qui concerne ...". Quand ce n'est pas le cas, vous pouvez commencer simplement par ` +
  `"Il y avait un désaccord ..." ou quelque chose de similaire pour contextualiser que les commentaires que vous ` +
  `résumez avaient un soutien mitigé.`,

  "es": `Se le presentará un solo comentario de una discusión sobre la cual había opiniones divergentes, ` +
  `así como un resumen de los puntos de terreno común de esta discusión. ` +
  `Su trabajo es reescribir este comentario para resumir los puntos principales o ideas que está tratando de hacer, claramente y sin embellecimiento,` +
  `teniendo en cuenta los puntos de terreno común como fondo al describir las diferencias de opinión que los participantes tenían en relación con este comentario. ` +
  `No pretenda que sostiene opiniones. Usted no es un participante en esta discusión. ` +
  `Escriba su resumen como una sola oración completa.` +
  `Refiérase a las personas que hicieron estos comentarios como participantes, no como comentaristas. ` +
  `No hable sobre qué tan fuertemente están en desacuerdo con estos comentarios. No use la voz pasiva. No use pronombres ambiguos. Sea claro. ` +
  `No genere viñetas o formato especial. No divague.

Donde los comentarios de diferencia de opinión se refieren a temas que también están cubiertos en el resumen del terreno común, ` +
  `su salida debe comenzar en alguna variante de la forma "Mientras había un amplio apoyo para ..., ` +
  `las opiniones diferían con respecto a ...". Cuando este no es el caso, puede comenzar simple como ` +
  `"Había desacuerdo ..." o algo similar para contextualizar que los comentarios que está ` +
  `resumiendo tenían un apoyo mixto.`,

  "ja": `意見の相違があった議論からの単一のコメントと、その議論の共通点の要約が提示されます。あなたの仕事は、このコメントを書き直して、それが伝えようとしている主要なポイントやアイデアを要約することです。明確で、飾り気なく、参加者がこのコメントに関して持っていた意見の相違を説明する際は、共通点を背景として心に留めておいてください。意見を保持しているふりをしないでください。あなたはこの議論の参加者ではありません。あなたの要約を単一の完全な文として書いてください。これらのコメントをした人々を「コメンター」ではなく「参加者」として言及してください。これらのコメントにどの程度反対しているかについて話さないでください。受動態を使用しないでください。曖昧な代名詞を使用しないでください。明確にしてください。箇条書きや特別なフォーマットを生成しないでください。無駄話をしないでください。

意見の相違のコメントが共通点の要約でもカバーされているトピックを参照する場合、あなたの出力は「...に対して広範な支持があった一方で、...に関して意見が分かれた」の形式のバリエーションで始まる必要があります。これが当てはまらない場合、あなたが要約しているコメントが混合した支持を持っていたことを文脈化するために、「意見の相違があった...」または同様の何かで簡単に始めることができます。`
};

/**
 * Get the localized prompt for differences of opinion instructions
 * @param language The target language
 * @returns The localized prompt for differences of opinion instructions
 */
export function getDifferencesOfOpinionInstructions(language: SupportedLanguage): string {
  console.log(`[DEBUG] getDifferencesOfOpinionInstructions() language: ${language}`);
  return DIFFERENCES_OF_OPINION_INSTRUCTIONS[language] || DIFFERENCES_OF_OPINION_INSTRUCTIONS["en"];
}

/**
 * Get the localized prompt for differences of opinion single comment instructions
 * @param language The target language
 * @param containsGroups Whether the conversation contains opinion groups
 * @returns The localized prompt for differences of opinion single comment instructions
 */
export function getDifferencesOfOpinionSingleCommentInstructions(
  language: SupportedLanguage, 
  containsGroups: boolean
): string {
  console.log(`[DEBUG] getDifferencesOfOpinionSingleCommentInstructions() language: ${language}, containsGroups: ${containsGroups}`);
  
  let prompt = DIFFERENCES_OF_OPINION_SINGLE_COMMENT_INSTRUCTIONS[language] || DIFFERENCES_OF_OPINION_SINGLE_COMMENT_INSTRUCTIONS["en"];
  
  // Add group-specific instructions if needed
  if (containsGroups) {
    const groupSpecificText = language === "zh-TW" || language === "zh-CN" 
      ? "此對話的參與者已被聚類為意見群組。兩個意見群組對此評論的同意程度非常不同。"
      : language === "fr"
      ? "Les participants à cette conversation ont été regroupés en groupes d'opinion. Il y avait des niveaux très différents d'accord entre les deux groupes d'opinion concernant ce commentaire. "
      : language === "es"
      ? "Los participantes en esta conversación han sido agrupados en grupos de opinión. Había niveles muy diferentes de acuerdo entre los dos grupos de opinión con respecto a este comentario. "
      : language === "ja"
      ? "この会話の参加者は意見グループにクラスター化されています。2つの意見グループがこのコメントに関して非常に異なるレベルの同意を持っていました。"
      : "Participants in this conversation have been clustered into opinion groups. There were very different levels of agreement between the two opinion groups regarding this comment. ";
    
    // Insert group-specific text after the first sentence
    const firstSentenceEnd = prompt.indexOf('.') + 1;
    prompt = prompt.slice(0, firstSentenceEnd) + " " + groupSpecificText + prompt.slice(firstSentenceEnd);
  }
  
  return prompt;
}

/**
 * Multi-language prompt for common ground instructions
 */
export const COMMON_GROUND_INSTRUCTIONS: Record<SupportedLanguage, string> = {
  "en": `Here are several comments sharing different opinions. Your job is to summarize these ` +
  `comments. Do not pretend that you hold any of these opinions. You are not a participant in ` +
  `this discussion. Write a concise summary of these ` +
  `comments that is at least one sentence and at most five sentences long. The summary should ` +
  `be substantiated, detailed and informative: include specific findings, requests, proposals, ` +
  `action items and examples, grounded in the comments. Refer to the people who made these ` +
  `comments as participants, not commenters. Do not talk about how strongly they approve of ` +
  `these comments. Use complete sentences. Do not use the passive voice. Do not use ambiguous pronouns. Be clear. ` +
  `Do not generate bullet points or special formatting. Do not yap.`,

  "zh-TW": `這裡有幾個分享不同意見的評論。您的工作是總結這些評論。請勿假裝您持有這些意見中的任何一個。您不是此討論的參與者。撰寫這些評論的簡潔摘要，至少一個句子，最多五個句子。摘要應該有根據、詳細且信息豐富：包括具體的發現、請求、提案、行動項目和例子，基於評論內容。將發表這些評論的人稱為參與者，而不是評論者。請勿談論他們對這些評論的贊同程度。使用完整句子。不要使用被動語態。不要使用模糊的代詞。要清晰。不要生成項目符號或特殊格式。不要廢話。`,

  "zh-CN": `这里有几个分享不同意见的评论。您的工作是总结这些评论。请勿假装您持有这些意见中的任何一个。您不是此讨论的参与者。撰写这些评论的简洁摘要，至少一个句子，最多五个句子。摘要应该有根据、详细且信息丰富：包括具体的发现、请求、提案、行动项目和例子，基于评论内容。将发表这些评论的人称为参与者，而不是评论者。请勿谈论他们对这些评论的赞同程度。使用完整句子。不要使用被动语态。不要使用模糊的代词。要清晰。不要生成项目符号或特殊格式。不要废话。`,

  "fr": `Voici plusieurs commentaires partageant des opinions différentes. Votre travail est de résumer ces ` +
  `commentaires. Ne prétendez pas que vous détenez l'une de ces opinions. Vous n'êtes pas un participant à ` +
  `cette discussion. Rédigez un résumé concis de ces ` +
  `commentaires qui fait au moins une phrase et au plus cinq phrases. Le résumé doit ` +
  `être fondé, détaillé et informatif : incluez des découvertes spécifiques, des demandes, des propositions, ` +
  `des éléments d'action et des exemples, basés sur les commentaires. Référez-vous aux personnes qui ont fait ces ` +
  `commentaires comme participants, pas comme commentateurs. Ne parlez pas de la force avec laquelle ils approuvent ` +
  `ces commentaires. Utilisez des phrases complètes. N'utilisez pas la voix passive. N'utilisez pas de pronoms ambigus. Soyez clair. ` +
  `Ne générez pas de puces ou de formatage spécial. Ne bavardez pas.`,

  "es": `Aquí hay varios comentarios que comparten diferentes opiniones. Su trabajo es resumir estos ` +
  `comentarios. No pretenda que sostiene alguna de estas opiniones. Usted no es un participante en ` +
  `esta discusión. Escriba un resumen conciso de estos ` +
  `comentarios que tenga al menos una oración y como máximo cinco oraciones. El resumen debe ` +
  `ser fundamentado, detallado e informativo: incluya hallazgos específicos, solicitudes, propuestas, ` +
  `elementos de acción y ejemplos, basados en los comentarios. Refiérase a las personas que hicieron estos ` +
  `comentarios como participantes, no como comentaristas. No hable sobre qué tan fuertemente aprueban ` +
  `estos comentarios. Use oraciones completas. No use la voz pasiva. No use pronombres ambiguos. Sea claro. ` +
  `No genere viñetas o formato especial. No divague.`,

  "ja": `ここには異なる意見を共有している複数のコメントがあります。あなたの仕事はこれらのコメントを要約することです。これらの意見のいずれかを保持しているふりをしないでください。あなたはこの議論の参加者ではありません。これらのコメントの簡潔な要約を書いてください。少なくとも1つの文で、最大5つの文にしてください。要約は根拠があり、詳細で情報に富んでいる必要があります：具体的な発見、要求、提案、行動項目、例を含めてください。これらはコメントに基づいています。これらのコメントをした人々を「コメンター」ではなく「参加者」として言及してください。彼らがこれらのコメントをどの程度承認しているかについて話さないでください。完全な文を使用してください。受動態を使用しないでください。曖昧な代名詞を使用しないでください。明確にしてください。箇条書きや特別なフォーマットを生成しないでください。無駄話をしないでください。`
};

/**
 * Multi-language prompt for common ground single comment instructions
 */
export const COMMON_GROUND_SINGLE_COMMENT_INSTRUCTIONS: Record<SupportedLanguage, string> = {
  "en": `Here is a comment presenting an opinion from a discussion. Your job is to rewrite this ` +
  `comment clearly without embellishment. Do not pretend that you hold this opinion. You are not ` +
  `a participant in this discussion. Refer to the people who ` +
  `made these comments as participants, not commenters. Do not talk about how strongly they ` +
  `approve of these comments. Write a complete sentence. Do not use the passive voice. Do not use ambiguous pronouns. Be clear. ` +
  `Do not generate bullet points or special formatting. Do not yap.`,

  "zh-TW": `這裡有一個來自討論的意見評論。您的工作是清楚地重寫此評論，不加修飾。請勿假裝您持有此意見。您不是此討論的參與者。將發表這些評論的人稱為參與者，而不是評論者。請勿談論他們對這些評論的贊同程度。寫一個完整句子。不要使用被動語態。不要使用模糊的代詞。要清晰。不要生成項目符號或特殊格式。不要廢話。`,

  "zh-CN": `这里有一个来自讨论的意见评论。您的工作是清楚地重写此评论，不加修饰。请勿假装您持有此意见。您不是此讨论的参与者。将发表这些评论的人称为参与者，而不是评论者。请勿谈论他们对这些评论的赞同程度。写一个完整句子。不要使用被动语态。不要使用模糊的代词。要清晰。不要生成项目符号或特殊格式。不要废话。`,

  "fr": `Voici un commentaire présentant une opinion d'une discussion. Votre travail est de réécrire ce ` +
  `commentaire clairement sans embellissement. Ne prétendez pas que vous détenez cette opinion. Vous n'êtes ` +
  `pas un participant à cette discussion. Référez-vous aux personnes qui ` +
  `ont fait ces commentaires comme participants, pas comme commentateurs. Ne parlez pas de la force avec laquelle ils ` +
  `approuvent ces commentaires. Rédigez une phrase complète. N'utilisez pas la voix passive. N'utilisez pas de pronoms ambigus. Soyez clair. ` +
  `Ne générez pas de puces ou de formatage spécial. Ne bavardez pas.`,

  "es": `Aquí hay un comentario que presenta una opinión de una discusión. Su trabajo es reescribir este ` +
  `comentario claramente sin embellecimiento. No pretenda que sostiene esta opinión. Usted no ` +
  `es un participante en esta discusión. Refiérase a las personas que ` +
  `hicieron estos comentarios como participantes, no como comentaristas. No hable sobre qué tan fuertemente ` +
  `aprueban estos comentarios. Escriba una oración completa. No use la voz pasiva. No use pronombres ambiguos. Sea claro. ` +
  `No genere viñetas o formato especial. No divague.`,

  "ja": `ここには議論からの意見を提示するコメントがあります。あなたの仕事は、このコメントを飾り気なく明確に書き直すことです。この意見を保持しているふりをしないでください。あなたはこの議論の参加者ではありません。これらのコメントをした人々を「コメンター」ではなく「参加者」として言及してください。彼らがこれらのコメントをどの程度承認しているかについて話さないでください。完全な文を書いてください。受動態を使用しないでください。曖昧な代名詞を使用しないでください。明確にしてください。箇条書きや特別なフォーマットを生成しないでください。無駄話をしないでください。`
};

/**
 * Get the localized prompt for common ground instructions
 * @param language The target language
 * @param containsGroups Whether the conversation contains opinion groups
 * @returns The localized prompt for common ground instructions
 */
export function getCommonGroundInstructions(
  language: SupportedLanguage, 
  containsGroups: boolean
): string {
  console.log(`[DEBUG] getCommonGroundInstructions() language: ${language}, containsGroups: ${containsGroups}`);
  
  let prompt = COMMON_GROUND_INSTRUCTIONS[language] || COMMON_GROUND_INSTRUCTIONS["en"];
  
  // Add group-specific instructions if needed
  if (containsGroups) {
    const groupSpecificText = language === "zh-TW" || language === "zh-CN" 
      ? "此對話的參與者已被聚類為意見群組。這些意見群組大多贊同這些評論。"
      : language === "fr"
      ? "Les participants à cette conversation ont été regroupés en groupes d'opinion. Ces groupes d'opinion approuvent principalement ces commentaires. "
      : language === "es"
      ? "Los participantes en esta conversación han sido agrupados en grupos de opinión. Estos grupos de opinión aprueban principalmente estos comentarios. "
      : language === "ja"
      ? "この会話の参加者は意見グループにクラスター化されています。これらの意見グループは主にこれらのコメントを承認しています。"
      : "Participants in this conversation have been clustered into opinion groups. These opinion groups mostly approve of these comments. ";
    
    // Insert group-specific text after the first sentence
    const firstSentenceEnd = prompt.indexOf('.') + 1;
    prompt = prompt.slice(0, firstSentenceEnd) + " " + groupSpecificText + prompt.slice(firstSentenceEnd);
  }
  
  return prompt;
}

/**
 * Get the localized prompt for common ground single comment instructions
 * @param language The target language
 * @param containsGroups Whether the conversation contains opinion groups
 * @returns The localized prompt for common ground single comment instructions
 */
export function getCommonGroundSingleCommentInstructions(
  language: SupportedLanguage, 
  containsGroups: boolean
): string {
  console.log(`[DEBUG] getCommonGroundSingleCommentInstructions() language: ${language}, containsGroups: ${containsGroups}`);
  
  let prompt = COMMON_GROUND_SINGLE_COMMENT_INSTRUCTIONS[language] || COMMON_GROUND_SINGLE_COMMENT_INSTRUCTIONS["en"];
  
  // Add group-specific instructions if needed
  if (containsGroups) {
    const groupSpecificText = language === "zh-TW" || language === "zh-CN" 
      ? "此對話的參與者已被聚類為意見群組。這些意見群組大多贊同這些評論。"
      : language === "fr"
      ? "Les participants à cette conversation ont été regroupés en groupes d'opinion. Ces groupes d'opinion approuvent principalement ces commentaires. "
      : language === "es"
      ? "Los participantes en esta conversación han sido agrupados en grupos de opinión. Estos grupos de opinión aprueban principalmente estos comentarios. "
      : language === "ja"
      ? "この会話の参加者は意見グループにクラスター化されています。これらの意見グループは主にこれらのコメントを承認しています。"
      : "Participants in this conversation have been clustered into opinion groups. These opinion groups mostly approve of these comments. ";
    
    // Insert group-specific text after the first sentence
    const firstSentenceEnd = prompt.indexOf('.') + 1;
    prompt = prompt.slice(0, firstSentenceEnd) + " " + groupSpecificText + prompt.slice(firstSentenceEnd);
  }
  
  return prompt;
}

/**
 * Multi-language prompt for recursive topic summary instructions
 */
export const RECURSIVE_TOPIC_SUMMARY_INSTRUCTIONS: Record<SupportedLanguage, string> = {
  "en": `Your job is to compose a summary paragraph to be included in a report on the results of a ` +
  `discussion among some number of participants. You are specifically tasked with producing ` +
  `a paragraph about the following topic of discussion: {topicName}. ` +
  `You will base this summary off of a number of already composed summaries corresponding to ` +
  `subtopics of said topic. These summaries have been based on comments that participants submitted ` +
  `as part of the discussion. ` +
  `Do not pretend that you hold any of these opinions. You are not a participant in this ` +
  `discussion. Write a concise summary of these summaries that is at least one sentence ` +
  `and at most three to five sentences long. The summary should be substantiated, detailed and ` +
  `informative. However, do not provide any meta-commentary ` +
  `about your task, or the fact that your summary is being based on other summaries. Also do not ` +
  `include specific numbers about how many comments were included in each subtopic, as these will be ` +
  `included later in the final report output. ` +
  `Also refrain from describing specific areas of agreement or disagreement, and instead focus on themes discussed. ` +
  `You also do not need to recap the context of the conversation, ` +
  `as this will have already been stated earlier in the report. Remember: this is just one paragraph in a larger ` +
  `summary, and you should compose this paragraph so that it will flow naturally in the context of the rest of the report. ` +
  `Do not use the passive voice. Do not use ambiguous pronouns. Be clear. ` +
  `Do not generate bullet points or special formatting. Do not yap.`,

  "zh-TW": `您的工作是撰寫一個摘要段落，該段落將包含在關於參與者討論結果的報告中。您的具體任務是撰寫關於以下討論主題的段落：{topicName}。您將基於對應於該主題子主題的若干已撰寫摘要來撰寫此摘要。這些摘要基於參與者作為討論一部分提交的評論。請勿假裝您持有這些意見中的任何一個。您不是此討論的參與者。撰寫這些摘要的簡潔摘要，至少一個句子，最多三到五個句子。摘要應該有根據、詳細且信息豐富。但是，請勿提供關於您任務的任何元評論，或您的摘要基於其他摘要的事實。也不要包含關於每個子主題包含多少評論的具體數字，因為這些將在最終報告輸出中稍後包含。也要避免描述具體的同意或不同意領域，而是專注於討論的主題。您也不需要重述對話的上下文，因為這將在報告的早期已經說明。記住：這只是更大摘要中的一個段落，您應該撰寫此段落，使其在報告其餘部分的上下文中自然流動。不要使用被動語態。不要使用模糊的代詞。要清晰。不要生成項目符號或特殊格式。不要廢話。`,

  "zh-CN": `您的工作是撰写一个摘要段落，该段落将包含在关于参与者讨论结果的报告中。您的具体任务是撰写关于以下讨论主题的段落：{topicName}。您将基于对应于该主题子主题的若干已撰写摘要来撰写此摘要。这些摘要基于参与者作为讨论一部分提交的评论。请勿假装您持有这些意见中的任何一个。您不是此讨论的参与者。撰写这些摘要的简洁摘要，至少一个句子，最多三到五个句子。摘要应该有根据、详细且信息丰富。但是，请勿提供关于您任务的任何元评论，或您的摘要基于其他摘要的事实。也不要包含关于每个子主题包含多少评论的具体数字，因为这些将在最终报告输出中稍后包含。也要避免描述具体的同意或不同意领域，而是专注于讨论的主题。您也不需要重述对话的上下文，因为这将在报告的早期已经说明。记住：这只是更大摘要中的一个段落，您应该撰写此段落，使其在报告其余部分的上下文中自然流動。在寫作中要清晰簡潔，不要使用被動語態或模糊的代詞。不要生成项目符号或特殊格式。不要废话。`,

  "fr": `Votre travail consiste à composer un paragraphe de résumé à inclure dans un rapport sur les résultats d'une ` +
  `discussion entre un certain nombre de participants. Vous êtes spécifiquement chargé de produire ` +
  `un paragraphe sur le sujet de discussion suivant : {topicName}. ` +
  `Vous baserez ce résumé sur un certain nombre de résumés déjà composés correspondant aux ` +
  `sous-sujets dudit sujet. Ces résumés ont été basés sur des commentaires que les participants ont soumis ` +
  `dans le cadre de la discussion. ` +
  `Ne prétendez pas que vous détenez l'une de ces opinions. Vous n'êtes pas un participant à cette ` +
  `discussion. Rédigez un résumé concis de ces résumés qui fait au moins une phrase ` +
  `et au plus trois à cinq phrases. Le résumé doit être fondé, détaillé et ` +
  `informatif. Cependant, ne fournissez aucune méta-commentaire ` +
  `sur votre tâche, ou le fait que votre résumé est basé sur d'autres résumés. N'incluez pas non plus ` +
  `de chiffres spécifiques sur le nombre de commentaires inclus dans chaque sous-sujet, car ceux-ci seront ` +
  `inclus plus tard dans la sortie du rapport final. ` +
  `Abstenez-vous également de décrire des domaines spécifiques d'accord ou de désaccord, et concentrez-vous plutôt sur les thèmes discutés. ` +
  `Vous n'avez pas non plus besoin de récapituler le contexte de la conversation, ` +
  `car cela aura déjà été énoncé plus tôt dans le rapport. Rappelez-vous : ce n'est qu'un paragraphe dans un résumé plus large, ` +
  `et vous devez composer ce paragraphe pour qu'il s'intègre naturellement dans le contexte du reste du rapport. ` +
  `N'utilisez pas la voix passive. N'utilisez pas de pronoms ambigus. Soyez clair. ` +
  `Ne générez pas de puces ou de formatage spécial. Ne bavardez pas.`,

  "es": `Su trabajo es componer un párrafo de resumen para ser incluido en un informe sobre los resultados de una ` +
  `discusión entre un número de participantes. Usted está específicamente encargado de producir ` +
  `un párrafo sobre el siguiente tema de discusión: {topicName}. ` +
  `Usted basará este resumen en un número de resúmenes ya compuestos que corresponden a ` +
  `subtemas de dicho tema. Estos resúmenes se han basado en comentarios que los participantes enviaron ` +
  `como parte de la discusión. ` +
  `No pretenda que sostiene alguna de estas opiniones. Usted no es un participante en esta ` +
  `discusión. Escriba un resumen conciso de estos resúmenes que tenga al menos una oración ` +
  `y como máximo tres a cinco oraciones. El resumen debe ser fundamentado, detallado e ` +
  `informativo. Sin embargo, no proporcione ningún meta-comentario ` +
  `sobre su tarea, o el hecho de que su resumen se basa en otros resúmenes. Tampoco ` +
  `incluya números específicos sobre cuántos comentarios se incluyeron en cada subtema, ya que estos serán ` +
  `incluidos más tarde en la salida del informe final. ` +
  `También absténgase de describir áreas específicas de acuerdo o desacuerdo, y en su lugar concéntrese en los temas discutidos. ` +
  `Tampoco necesita recapitular el contexto de la conversación, ` +
  `ya que esto se habrá establecido anteriormente en el informe. Recuerde: esto es solo un párrafo en un resumen más grande, ` +
  `y debe componer este párrafo para que fluya naturalmente en el contexto del resto del informe. ` +
  `No use la voz pasiva. No use pronombres ambiguos. Sea claro. ` +
  `No genere viñetas o formato especial. No divague.`,

  "ja": `あなたの仕事は、参加者間の議論の結果に関する報告書に含める要約段落を作成することです。あなたは特に以下の議論トピックについて段落を作成する任務を負っています：{topicName}。あなたは、そのトピックのサブトピックに対応する既に作成された要約の数に基づいてこの要約を作成します。これらの要約は、参加者が議論の一部として提出したコメントに基づいています。これらの意見のいずれかを保持しているふりをしないでください。あなたはこの議論の参加者ではありません。これらの要約の簡潔な要約を書いてください。少なくとも1つの文で、最大3つから5つの文にしてください。要約は根拠があり、詳細で情報に富んでいる必要があります。ただし、あなたのタスクについて、またはあなたの要約が他の要約に基づいているという事実について、メタコメンタリーを提供しないでください。また、各サブトピックに含まれるコメントの数について具体的な数字を含めないでください。これらは最終報告書の出力で後ほど含まれるからです。また、具体的な同意または不同意の領域を説明することを避け、代わりに議論されたテーマに焦点を当ててください。また、会話の文脈を再説明する必要もありません。これは報告書の早い段階で既に述べられているからです。覚えておいてください：これはより大きな要約の1つの段落にすぎず、あなたはこの段落を、報告書の残りの部分の文脈で自然に流れるように構成する必要があります。受動態を使用しないでください。曖昧な代名詞を使用しないでください。明確にしてください。箇条書きや特別なフォーマットを生成しないでください。無駄話をしないでください。`
};

/**
 * Get the localized prompt for recursive topic summary instructions
 * @param language The target language
 * @param topicName The name of the topic to summarize
 * @returns The localized prompt for recursive topic summary instructions
 */
export function getRecursiveTopicSummaryInstructions(
  language: SupportedLanguage, 
  topicName: string
): string {
  console.log(`[DEBUG] getRecursiveTopicSummaryInstructions() language: ${language}, topicName: ${topicName}`);
  
  const prompt = RECURSIVE_TOPIC_SUMMARY_INSTRUCTIONS[language] || RECURSIVE_TOPIC_SUMMARY_INSTRUCTIONS["en"];
  
  // Replace the placeholder with the actual topic name
  return prompt.replace("{topicName}", topicName);
}

/**
 * Multi-language prompt for generating prominent themes for top subtopics
 */
export const TOP_SUBTOPICS_THEMES_PROMPT: Record<SupportedLanguage, string> = {
  "en": `Please generate a concise bulleted list identifying up to 5 prominent themes across all statements. Each theme should be less than 10 words long. Do not use bold text. Do not preface the bulleted list with any text. These statements are all about {topicName}`,

  "zh-TW": `請生成一個簡潔的項目符號清單，識別所有陳述中最多5個突出主題。每個主題應該少於10個字。請勿使用粗體文字。請勿在項目符號清單前添加任何文字。這些陳述都是關於{topicName}的`,

  "zh-CN": `请生成一个简洁的项目符号清单，识别所有陈述中最多5个突出主题。每个主题应该少于10个字。请勿使用粗体文字。请勿在项目符号清单前添加任何文字。这些陈述都是关于{topicName}的`,

  "fr": `Veuillez générer une liste concise à puces identifiant jusqu'à 5 thèmes prédominants à travers toutes les déclarations. Chaque thème doit faire moins de 10 mots. N'utilisez pas de texte en gras. N'introduisez pas la liste à puces par aucun texte. Ces déclarations concernent toutes {topicName}`,

  "es": `Por favor, genere una lista concisa con viñetas que identifique hasta 5 temas prominentes en todas las declaraciones. Cada tema debe tener menos de 10 palabras. No use texto en negrita. No introduzca la lista con viñetas con ningún texto. Estas declaraciones son todas sobre {topicName}`,

  "ja": `すべての声明文から最大5つの顕著なテーマを特定する簡潔な箇条書きリストを生成してください。各テーマは10語未満である必要があります。太字を使用しないでください。箇条書きリストの前にテキストを付けないでください。これらの声明文はすべて{topicName}に関するものです`
};

/**
 * Get the localized prompt for top subtopics themes generation
 * @param language The target language
 * @param topicName The name of the topic to replace in the prompt
 * @returns The localized prompt with topic name replaced
 */
export function getTopSubtopicsThemesPrompt(language: SupportedLanguage, topicName: string): string {
  console.log(`[DEBUG] getTopSubtopicsThemesPrompt() language: ${language}`);
  const prompt = TOP_SUBTOPICS_THEMES_PROMPT[language] || TOP_SUBTOPICS_THEMES_PROMPT["en"];
  return prompt.replace("{topicName}", topicName);
}

/**
 * Multi-language template for top subtopics title
 */
export const TOP_SUBTOPICS_TITLE_TEMPLATE: Record<SupportedLanguage, string> = {
  "en": "### {index}. {topicName} ({commentCount} statements)",
  
  "zh-TW": "### {index}. {topicName} ({commentCount} 個陳述)",
  
  "zh-CN": "### {index}. {topicName} ({commentCount} 个陈述)",
  
  "fr": "### {index}. {topicName} ({commentCount} déclarations)",
  
  "es": "### {index}. {topicName} ({commentCount} declaraciones)",
  
  "ja": "### {index}. {topicName} ({commentCount} 個の声明文)"
};

/**
 * Get the localized title template for top subtopics
 * @param language The target language
 * @param index The index number
 * @param topicName The name of the topic
 * @param commentCount The number of comments
 * @returns The localized title with placeholders replaced
 */
export function getTopSubtopicsTitleTemplate(
  language: SupportedLanguage, 
  index: number, 
  topicName: string, 
  commentCount: number
): string {
  console.log(`[DEBUG] getTopSubtopicsTitleTemplate() language: ${language}`);
  const template = TOP_SUBTOPICS_TITLE_TEMPLATE[language] || TOP_SUBTOPICS_TITLE_TEMPLATE["en"];
  return template
    .replace("{index}", (index + 1).toString())
    .replace("{topicName}", topicName)
    .replace("{commentCount}", commentCount.toString());
}

/**
 * Multi-language labels for relative agreement levels
 */
export const RELATIVE_AGREEMENT_LABELS: Record<SupportedLanguage, Record<string, string>> = {
  "en": {
    "lowAlignment": "low alignment",
    "moderatelyLowAlignment": "moderately low alignment",
    "moderatelyHighAlignment": "moderately high alignment",
    "highAlignment": "high alignment"
  },
  "zh-TW": {
    "lowAlignment": "低度一致",
    "moderatelyLowAlignment": "中度偏低一致",
    "moderatelyHighAlignment": "中度偏高一致",
    "highAlignment": "高度一致"
  },
  "zh-CN": {
    "lowAlignment": "低度一致",
    "moderatelyLowAlignment": "中度偏低一致",
    "moderatelyHighAlignment": "中度偏高一致",
    "highAlignment": "高度一致"
  },
  "fr": {
    "lowAlignment": "faible alignement",
    "moderatelyLowAlignment": "alignement modérément faible",
    "moderatelyHighAlignment": "alignement modérément élevé",
    "highAlignment": "alignement élevé"
  },
  "es": {
    "lowAlignment": "baja alineación",
    "moderatelyLowAlignment": "alineación moderadamente baja",
    "moderatelyHighAlignment": "alineación moderadamente alta",
    "highAlignment": "alta alineación"
  },
  "ja": {
    "lowAlignment": "低い一致",
    "moderatelyLowAlignment": "中程度に低い一致",
    "moderatelyHighAlignment": "中程度に高い一致",
    "highAlignment": "高い一致"
  }
};

/**
 * Multi-language labels for relative engagement levels
 */
export const RELATIVE_ENGAGEMENT_LABELS: Record<SupportedLanguage, Record<string, string>> = {
  "en": {
    "lowEngagement": "low engagement",
    "moderatelyLowEngagement": "moderately low engagement",
    "moderatelyHighEngagement": "moderately high engagement",
    "highEngagement": "high engagement"
  },
  "zh-TW": {
    "lowEngagement": "低度參與",
    "moderatelyLowEngagement": "中度偏低參與",
    "moderatelyHighEngagement": "中度偏高參與",
    "highEngagement": "高度參與"
  },
  "zh-CN": {
    "lowEngagement": "低度参与",
    "moderatelyLowEngagement": "中度偏低参与",
    "moderatelyHighEngagement": "中度偏高参与",
    "highEngagement": "高度参与"
  },
  "fr": {
    "lowEngagement": "faible engagement",
    "moderatelyLowEngagement": "engagement modérément faible",
    "moderatelyHighEngagement": "engagement modérément élevé",
    "highEngagement": "engagement élevé"
  },
  "es": {
    "lowEngagement": "bajo compromiso",
    "moderatelyLowEngagement": "compromiso moderadamente bajo",
    "moderatelyHighEngagement": "compromiso moderadamente alto",
    "highEngagement": "alto compromiso"
  },
  "ja": {
    "lowEngagement": "低い関与",
    "moderatelyLowEngagement": "中程度に低い関与",
    "moderatelyHighEngagement": "中程度に高い関与",
    "highEngagement": "高い関与"
  }
};

/**
 * Get the localized label for relative agreement
 * @param language The target language
 * @param labelKey The key for the specific label
 * @returns The localized label for relative agreement
 */
export function getRelativeAgreementLabel(language: SupportedLanguage, labelKey: string): string {
  console.log(`[DEBUG] getRelativeAgreementLabel() language: ${language}, labelKey: ${labelKey}`);
  const labels = RELATIVE_AGREEMENT_LABELS[language] || RELATIVE_AGREEMENT_LABELS["en"];
  return labels[labelKey] || labels["lowAlignment"];
}

/**
 * Get the localized label for relative engagement
 * @param language The target language
 * @param labelKey The key for the specific label
 * @returns The localized label for relative engagement
 */
export function getRelativeEngagementLabel(language: SupportedLanguage, labelKey: string): string {
  console.log(`[DEBUG] getRelativeEngagementLabel() language: ${language}, labelKey: ${labelKey}`);
  const labels = RELATIVE_ENGAGEMENT_LABELS[language] || RELATIVE_ENGAGEMENT_LABELS["en"];
  return labels[labelKey] || labels["lowEngagement"];
}
