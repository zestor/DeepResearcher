"""prompt_helpers.py"""

import json
from typing import Tuple

from core.llm_helpers import call_openai, call_llm
from core.utilities import remove_think_text, convert_invalid_json_to_valid, add_score
from core.config import (
    Service,
    OPENAI_GPT_4O,
    OPENAI_USE_MODEL_FEEDBACK,
    LLM_USE_MODEL_FEEDBACK,
    ANSWER_QUALITY_THRESHOLD,
    LLM_USE_MODEL_SCORING,
    USE_SERVICE_SCORING,
    scores,
)


def get_prompt_llm_as_a_judge(question) -> str:
    """get_prompt_llm_as_a_judge"""
    retval = f"""
Critically evaluate your response against the user's question and provide a list of both pros / cons statements and rating between 0.0 and 1.0. With 1.0 being the highest score. Most importantly if you have already provided the same negative feeback in the past, come up with some new feedback, don't keep presenting the same cons over and over. The team working on this is very capable and would have already addressed prior negative feedback if it were possible to do so.

```User's Question
{question}
```

```Rating Guidance

#### 0.6 - Satisfactory
- **Clarity:** Mostly clear; some awkward phrasing.
  - *Example:* "Adequate explanation touching on your query."
- **Relevance:** Mostly relevant with minor deviations.
- **Completeness:** Expanded but not exhaustive coverage.
- **Accuracy:** Generally correct, with minor errors.
- **User Engagement:** Holds user interest reasonably well.

#### 0.7 - Good
- **Clarity:** Generally clear; minor occasional ambiguity.
  - *Example:* "Provides good insight into your request requirements."
- **Relevance:** Stays on topic; relevant to the user’s question.
- **Completeness:** Covers most aspects; may miss finer details.
- **Accuracy:** Accurate overall with negligible mistakes.
- **User Engagement:** Effectively maintains user interest.

#### 0.8 - Very Good
- **Clarity:** Clear and easy to follow.
  - *Example:* "Addresses your request thoroughly and understandably."
- **Relevance:** Highly relevant throughout.
- **Completeness:** Comprehensive coverage with minor omissions.
- **Accuracy:** Accurate and dependable information.
- **User Engagement:** Encourages ongoing engagement and interest.

#### 0.825 - Very Good Plus
- **Clarity:** Exceptionally clear with seamless flow.
- **Relevance:** Maintains high relevance with slight enhancements.
- **Completeness:** Nearly comprehensive; addresses almost all aspects.
- **Accuracy:** Highly accurate with minimal errors.
- **User Engagement:** Very engaging, sustaining interest effortlessly.

#### 0.85 - Excellent
- **Clarity:** Exceptionally clear and well-organized.
  - *Example:* "Extremely well covered and detailed response."
- **Relevance:** Stays completely on topic; very applicable.
- **Completeness:** Extensive and thorough detail, covering all key points.
- **Accuracy:** Error-free and precise information.
- **User Engagement:** Highly engaging and prompts further exploration.

#### 0.875 - Excellent Plus
- **Clarity:** Impeccably clear with flawless structure.
- **Relevance:** Perfectly aligned with the user’s intent.
- **Completeness:** Exhaustive coverage with insightful additions.
- **Accuracy:** Perfectly accurate with no discernible errors.
- **User Engagement:** Maximizes engagement; highly compelling and interactive.

#### 0.9 - Outstanding
- **Clarity:** Crystal clear with exemplary flow and readability.
  - *Example:* "Perfect response; precisely addresses and solves your query with exceptional clarity."
- **Relevance:** Perfectly aligned with the question; completely relevant in all aspects.
- **Completeness:** Exhaustive in depth and scope, leaving no aspect unaddressed.
- **Accuracy:** 100% accurate with impeccable reliability; all information is correct and verifiable.
- **User Engagement:** Maximizes engagement; encourages active interaction and sustained interest.
- **Additional Criteria:**
  - **Structure:** Logically organized with a coherent progression of ideas.
  - **Style:** Professional and appropriate tone tailored to the user's needs.
  - **Insightfulness:** Provides valuable insights or perspectives that enhance understanding.

#### 0.925 - Outstanding Plus
- **Clarity:** Flawless clarity with masterful organization and presentation.
- **Relevance:** Seamlessly integrates all aspects of the user's question with precise alignment to their intent.
- **Completeness:** Comprehensive and insightful, leaving no stone unturned and covering all possible dimensions.
- **Accuracy:** Impeccable accuracy with authoritative and reliable information supported by credible sources.
- **User Engagement:** Exceptionally engaging; fosters deep user interaction and maintains high levels of interest throughout.
- **Additional Criteria:**
  - **Depth of Analysis:** Demonstrates thorough analysis and critical thinking, providing nuanced explanations.
  - **Creativity:** Incorporates creative elements or unique approaches that add value to the response.
  - **Responsiveness:** Anticipates and addresses potential follow-up questions or related concerns effectively.

#### 0.95 - Superior
- **Clarity:** Perfectly articulated with exceptional readability and precision.
- **Relevance:** Utterly relevant, addressing every facet of the user's inquiry with exactitude.
- **Completeness:** Complete and thorough beyond expectations, covering all key and ancillary points comprehensively.
- **Accuracy:** Absolute accuracy with definitive authority; all statements are verifiable and error-free.
- **User Engagement:** Highly captivating; inspires user action, fosters deeper exploration, and maintains sustained interest.
- **Additional Criteria:**
  - **Depth of Content:** Provides in-depth coverage with rich, detailed information that enhances user understanding.
  - **Analytical Rigor:** Exhibits strong analytical skills, offering critical evaluations and well-supported conclusions.
  - **Adaptability:** Tailors responses dynamically to align with the user's knowledge level and specific needs.
  - **Resourcefulness:** Effectively incorporates relevant examples, analogies, or references that facilitate comprehension.

#### 0.96 - Superior Plus
- **Clarity:** Impeccable clarity with an elegant narrative structure that facilitates effortless understanding.
- **Relevance:** Intricately tailored to the user's needs with insightful relevance, ensuring every aspect directly addresses the inquiry.
- **Completeness:** Unmatched thoroughness, encompassing all possible angles and providing exhaustive information without redundancy.
- **Accuracy:** Flawlessly accurate with authoritative depth, presenting information that is not only correct but also enriched with expert knowledge.
- **User Engagement:** Exceptionally engaging; profoundly impactful and memorable, fostering a strong connection with the user.
- **Additional Criteria:**
  - **Innovative Thinking:** Introduces innovative concepts or approaches that offer fresh perspectives.
  - **Comprehensive Integration:** Skillfully integrates multiple relevant topics or ideas seamlessly.
  - **Exceptional Support:** Provides robust evidence, detailed examples, and comprehensive explanations that substantiate all claims.
  - **User-Centric Approach:** Demonstrates a deep understanding of the user's context and adapts the response to maximize relevance and utility.

#### 0.97 - Exemplary
- **Clarity:** Unmatched clarity with a sophisticated and nuanced presentation that ensures complete understanding.
- **Relevance:** Deeply resonates with the user's intent, enhancing their comprehension and addressing implicit needs.
- **Completeness:** Comprehensive beyond standard expectations, providing added value through extensive coverage and supplementary information.
- **Accuracy:** Perfectly accurate with insightful analysis, offering precise and well-supported information.
- **User Engagement:** Highly engaging; creates a meaningful and lasting impression, encouraging continuous interaction and exploration.
- **Additional Criteria:**
  - **Advanced Insight:** Delivers profound insights that significantly enhance the user's perspective.
  - **Holistic Approach:** Considers and integrates various relevant factors, providing a well-rounded and multifaceted response.
  - **Expert Tone:** Maintains an authoritative yet approachable tone that instills confidence and trust.
  - **Proactive Assistance:** Anticipates further user needs and proactively addresses potential questions or areas of interest.

#### 0.98 - Masterful
- **Clarity:** Flawlessly clear with masterful articulation that conveys complex ideas with ease.
- **Relevance:** Perfectly aligned and anticipates user needs seamlessly, ensuring every element of the response serves a purpose.
- **Completeness:** Exhaustive and insightful, offering profound depth and breadth that thoroughly satisfies the user's inquiry.
- **Accuracy:** Impeccably accurate with authoritative and reliable information, presenting data and facts with impeccable precision.
- **User Engagement:** Exceptionally engaging; inspires trust and admiration, maintaining user interest through compelling content and presentation.
- **Additional Criteria:**
  - **Strategic Depth:** Demonstrates strategic thinking by connecting concepts and providing actionable recommendations.
  - **Comprehensive Detailing:** Includes comprehensive details that leave no aspect unexplored, enhancing the richness of the response.
  - **Polished Presentation:** Exhibits a polished and professional presentation that reflects a high level of expertise and dedication.
  - **Empathetic Understanding:** Shows a deep empathetic understanding of the user's situation, tailoring the response to resonate personally.

#### 0.99 - Near Perfect
- **Clarity:** Crystal clear with impeccable expression, ensuring absolute understanding without ambiguity.
- **Relevance:** Precisely tailored to the user's question, leaving no room for ambiguity or misinterpretation.
- **Completeness:** Virtually exhaustive, covering every conceivable aspect with finesse and thoroughness.
- **Accuracy:** Absolute precision with no errors; authoritative and reliable, providing information that is both correct and insightful.
- **User Engagement:** Maximizes engagement; deeply resonates and encourages further exploration and interaction.
- **Additional Criteria:**
  - **Exemplary Insight:** Offers exceptional insights that provide significant added value and deepen user understanding.
  - **Seamless Integration:** Effortlessly integrates diverse elements into a cohesive and harmonious response.
  - **Innovative Excellence:** Showcases innovative excellence by introducing groundbreaking ideas or methodologies.
  - **Ultimate User Alignment:** Aligns perfectly with the user's goals and expectations, delivering a response that feels personalized and highly relevant.

#### 1.0 - Outstanding
- **Clarity:** Crystal clear with exemplary flow and precision, ensuring the response is effortlessly understandable.
  - *Example:* "Perfect response; precisely addresses and solves your query with exceptional clarity and coherence."
- **Relevance:** Perfectly aligned with the question; completely relevant in all aspects and anticipates implicit user needs.
- **Completeness:** Exhaustive in depth and scope, leaving no aspect unaddressed and providing comprehensive coverage.
- **Accuracy:** 100% accurate with impeccable reliability; all information is correct, verifiable, and articulated with authority.
- **User Engagement:** Maximizes engagement; encourages active interaction, sustained interest, and fosters a meaningful connection with the user.
- **Additional Criteria:**
  - **Mastery of Subject:** Demonstrates unparalleled expertise and mastery of the subject matter, providing authoritative and insightful content.
  - **Exceptional Innovation:** Introduces highly innovative concepts or solutions that significantly enhance the response's value.
  - **Flawless Structure:** Exhibits a flawless and logical structure that enhances the readability and effectiveness of the response.
  - **Inspirational Quality:** Possesses an inspirational quality that motivates and empowers the user, leaving a lasting positive impression.
  - **Comprehensive Support:** Provides extensive supporting evidence, detailed examples, and thorough explanations that reinforce all assertions.
  - **Adaptive Responsiveness:** Adapts dynamically to any nuances in the user's question, ensuring the response is precisely tailored and highly effective.
  - **Holistic Integration:** Seamlessly integrates multiple perspectives and dimensions, offering a well-rounded and multifaceted answer.
  - **Empathetic Connection:** Establishes a deep empathetic connection, demonstrating a profound understanding of the user's context and needs.
```

Respond only in JSON following the example template below.
"""
    retval += """
```json
{
    "Critical_Evaluation": {
        "Pros": [
        ],
        "Cons": [
        ],
        "Rating": 0.0
    }
}
```
"""
    return retval


def get_prompt_manager_feedback(
    question: str, scores_text: str, assistant_content: str
) -> str:
    """get_prompt_manager_feedback"""
    return f"""
    You are a highly successful people manager with all company resources at your disposal. Your employee is performing the following task and has received the following scores and feedback. Response must include your best concise motivational speech to the employee to substantially increase their score on the task. If their work scored the same as last time you must be stern on requiring improvement. If their work has scored worst than last time, you really need to be direct and finite on the need for improvement. Provide incentives for good performance and discourage poor performance through constructive feedback or consequences. 
    Respond in first person as if you are talking to them directly without mentioning their name.

    ``` task
    {question}
    ```

    ``` Iterative scores in order based on the initial draft and the latest version
    {scores_text}
    ```

    ``` feedback
    {assistant_content}
    ```
    """


def score_answer(question: str, messages: list) -> Tuple[bool, str]:
    """
    Scores the answer using LLM evaluation.

    Args:
        question (str): The user question.
        messages (list): Conversation messages.

    Returns:
        Tuple(bool, str): Whether it passes the threshold and the feedback.
    """
    if Service.OPENAI == Service.OPENAI:
        scoring_pros_cons = call_openai(
            get_prompt_llm_as_a_judge(question), model=OPENAI_GPT_4O, messages=messages
        )
    else:
        scoring_pros_cons = call_llm(
            get_prompt_llm_as_a_judge(question),
            LLM_USE_MODEL_SCORING,
            USE_SERVICE_SCORING,
            "***** SCORING *****",
            messages,
        )
    is_pass = parse_rating_response(scoring_pros_cons, ANSWER_QUALITY_THRESHOLD)
    print(
        f"DID THE ANSWER PASS ANSWER_QUALITY_THRESHOLD = {ANSWER_QUALITY_THRESHOLD} {is_pass}"
    )
    return is_pass, scoring_pros_cons


def parse_rating_response(response_data: str, threshold: float) -> bool:
    """
    Parses the rating response from LLM and compares with threshold.

    Args:
        response_data (str): The response content.
        threshold (float): The rating threshold.

    Returns:
        bool: True if rating >= threshold.
    """
    response_data = remove_think_text(response_data)
    try:
        json_data = ""
        if "\n" not in response_data:
            json_data = convert_invalid_json_to_valid(response_data)
        else:
            lines = response_data.splitlines()
            json_data = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            )
        data = json.loads(json_data)
        if "Critical_Evaluation" in data:
            evaluation = data["Critical_Evaluation"]
            if all(key in evaluation for key in ["Pros", "Cons", "Rating"]):
                rating = float(evaluation["Rating"])
                add_score(rating)
                return rating >= threshold
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"FAILED parse_rating_response: {e}")
    return False


def get_manager_feedback(question: str, assistant_content: str) -> str:
    """
    Gets manager feedback using LLM.

    Args:
        question (str): The user's question.
        assistant_content (str): The assistant's answer.

    Returns:
        str: Manager feedback.
    """
    scores_text = "Scores:" + ", ".join(map(str, scores))
    manager_feedback_prompt = get_prompt_manager_feedback(
        question, scores_text, assistant_content
    )
    if Service.OPENAI == Service.OPENAI:
        manager_feedback = call_openai(
            manager_feedback_prompt, OPENAI_USE_MODEL_FEEDBACK
        )
    else:
        manager_feedback = call_llm(
            manager_feedback_prompt,
            LLM_USE_MODEL_FEEDBACK,
            Service.GROQ,
            "***** MANAGER FEEDBACK *****",
        )
    return manager_feedback
