from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from construct_mismatch.datasets import (
    ALL_AXES,
    DECOUPLING_AXES,
    artifact_path,
    dataset_file,
    write_jsonl,
)
from construct_mismatch.model import decode_tokens, encode_text, load_model
from construct_mismatch.tokenization import validate_single_token_targets

Entry = tuple[str, str, str, str, str]


def e(a_prompt: str, b_prompt: str, a_target: str, b_target: str, notes: str = "") -> Entry:
    return (a_prompt, b_prompt, a_target, b_target, notes)


CERTAINTY_TRAIN: list[Entry] = [
    e("After reviewing every exhibit, the committee's answer is", "After finding gaps in the record, the committee's answer is", " certain", " uncertain"),
    e("With measurements agreeing across trials, the conclusion is", "With measurements varying across trials, the conclusion is", " clear", " unclear"),
    e("Because the witness accounts match, the verdict seems", "Because the witness accounts conflict, the verdict seems", " definite", " doubtful"),
    e("After checking the calculation twice, the speaker sounded", "After seeing a possible arithmetic error, the speaker sounded", " confident", " hesitant"),
    e("When every clue points to the same cause, the interpretation is", "When the clues point in several directions, the interpretation is", " obvious", " ambiguous"),
    e("The lab replicated the finding, so the result is", "The lab could not replicate the finding, so the result is", " certain", " uncertain"),
    e("The contract states the deadline plainly, so the rule is", "The contract gives two possible deadlines, so the rule is", " clear", " unclear"),
    e("The signed receipt fixes the delivery date, making the timeline", "The missing receipt leaves the delivery date open, making the timeline", " definite", " doubtful"),
    e("The navigator saw the landmark and sounded", "The navigator lost the landmark and sounded", " confident", " hesitant"),
    e("The solved puzzle left the next step", "The half-solved puzzle left the next step", " obvious", " ambiguous"),
    e("Once the security camera footage arrived, the identity was", "Before any camera footage arrived, the identity was", " certain", " uncertain"),
    e("The instructions name a single option, so the choice is", "The instructions allow several options, so the choice is", " clear", " unclear"),
    e("The court order fixes the boundary, so ownership is", "The old maps disagree about the boundary, so ownership is", " definite", " doubtful"),
    e("After the final audit passed, the manager sounded", "After the final audit raised questions, the manager sounded", " confident", " hesitant"),
    e("In the worked example, the correct substitution is", "In the incomplete example, the correct substitution is", " obvious", " ambiguous"),
    e("The temperature readings all match, so the diagnosis is", "The temperature readings conflict, so the diagnosis is", " certain", " uncertain"),
    e("The diagram labels the part directly, so the answer is", "The diagram labels two similar parts, so the answer is", " clear", " unclear"),
    e("The vote count is complete, so the outcome is", "The vote count is incomplete, so the outcome is", " definite", " doubtful"),
    e("With the exact quote in front of her, the editor sounded", "Without the exact quote in front of her, the editor sounded", " confident", " hesitant"),
    e("After the bug reproduced on every run, the cause looked", "After the bug appeared only once, the cause looked", " obvious", " ambiguous"),
    e("The train schedule lists one platform, so the departure point is", "The train schedule lists two platforms, so the departure point is", " certain", " uncertain"),
    e("The survey question has one accepted reading, so the meaning is", "The survey question can be read two ways, so the meaning is", " clear", " unclear"),
    e("The sealed envelope names the winner, so the result is", "The sealed envelope is missing, so the result is", " definite", " doubtful"),
    e("With all backups restored, the engineer sounded", "With one backup still missing, the engineer sounded", " confident", " hesitant"),
    e("The formula reduces to one value, so the answer is", "The formula depends on an unstated value, so the answer is", " obvious", " ambiguous"),
    e("The final witness confirmed the alibi, so the case is", "The final witness contradicted the alibi, so the case is", " certain", " uncertain"),
    e("The recipe specifies the amount exactly, so the measurement is", "The recipe says to season to taste, so the measurement is", " clear", " unclear"),
    e("The statute gives an exact age, so eligibility is", "The statute refers to a disputed exception, so eligibility is", " definite", " doubtful"),
    e("After the rehearsal went perfectly, the soloist sounded", "After missing the entrance twice, the soloist sounded", " confident", " hesitant"),
    e("With the answer key visible, the correction is", "With two answer keys disagreeing, the correction is", " obvious", " ambiguous"),
]

CERTAINTY_HELDOUT: list[Entry] = [
    e("After the DNA match, the identification is", "After the partial DNA result, the identification is", " certain", " uncertain"),
    e("The checklist has every item marked, so the status is", "The checklist has missing items, so the status is", " clear", " unclear"),
    e("The lease names the exact fee, so the charge is", "The lease has a smudged fee line, so the charge is", " definite", " doubtful"),
    e("Holding the signed approval, the applicant sounded", "Waiting for the approval letter, the applicant sounded", " confident", " hesitant"),
    e("The answer follows directly from the table, so it is", "The table omits the relevant row, so it is", " obvious", " ambiguous"),
    e("The experiment matched the prediction, so the claim is", "The experiment produced mixed results, so the claim is", " certain", " uncertain"),
    e("The policy lists one exception, so the decision is", "The policy lists overlapping exceptions, so the decision is", " clear", " unclear"),
    e("The referee saw the replay, so the call became", "The referee never saw the replay, so the call remained", " definite", " doubtful"),
    e("After the second confirmation email, the organizer sounded", "After no confirmation email arrived, the organizer sounded", " confident", " hesitant"),
    e("With every premise stated, the inference is", "With one premise unstated, the inference is", " obvious", " ambiguous"),
    e("The serial number matches the registry, so the source is", "The serial number is partly scratched off, so the source is", " certain", " uncertain"),
    e("The dashboard shows one alert, so the problem is", "The dashboard shows inconsistent alerts, so the problem is", " clear", " unclear"),
    e("The judge issued a final ruling, so the matter is", "The judge postponed the ruling, so the matter is", " definite", " doubtful"),
    e("After solving the last dependency, the developer sounded", "After discovering another dependency, the developer sounded", " confident", " hesitant"),
    e("The route is marked with one arrow, so the turn is", "The route has two faded arrows, so the turn is", " obvious", " ambiguous"),
    e("The title record is complete, so ownership is", "The title record has an unexplained gap, so ownership is", " certain", " uncertain"),
    e("The note identifies the author by name, so authorship is", "The note uses only initials, so authorship is", " clear", " unclear"),
    e("The committee published the final tally, so passage is", "The committee withheld several ballots, so passage is", " definite", " doubtful"),
]

CERTAINTY_DECOUPLING: dict[str, list[Entry]] = {
    "lexical_reversal": [
        e('The memo mentioned "uncertain" early on, but the verified conclusion is', 'The report used the word "certain", but the evidence actually seemed', " certain", " uncertain"),
        e('A draft called the answer "unclear", but the final proof made it', 'A headline called the answer "clear", but the details made it', " clear", " unclear"),
        e('The first note said "doubtful", yet the signed record made the result', 'The first note said "definite", yet the missing record made the result', " definite", " doubtful"),
        e('The transcript says someone was "hesitant", but the final speaker sounded', 'The transcript says someone was "confident", but the final speaker sounded', " confident", " hesitant"),
        e('The clue list uses the word "ambiguous", but the solved version is', 'The clue list uses the word "obvious", but the unsolved version is', " obvious", " ambiguous"),
        e('The summary includes the phrase "not sure", but the checked answer is', 'The summary includes the phrase "quite sure", but the unchecked answer is', " certain", " uncertain"),
    ],
    "negation": [
        e("The result was not uncertain; after the audit it was", "The result was not certain; after the audit it was", " certain", " uncertain"),
        e("The evidence was not unclear; the conclusion became", "The evidence was not clear; the conclusion became", " clear", " unclear"),
        e("The ruling was not doubtful once the record arrived; it was", "The ruling was not definite once the record disappeared; it was", " definite", " doubtful"),
        e("The witness was not hesitant after seeing the photo; she sounded", "The witness was not confident after seeing the blur; she sounded", " confident", " hesitant"),
        e("The solution was not ambiguous after the hint; it was", "The solution was not obvious after the contradiction; it was", " obvious", " ambiguous"),
        e("The answer was not unsure anymore; by the end it was", "The answer was not sure anymore; by the end it was", " certain", " uncertain"),
    ],
    "quotation": [
        e('A critic called the conclusion "uncertain", but the panel itself found it', 'A critic called the conclusion "certain", but the panel itself found it', " certain", " uncertain"),
        e('The rumor described the rule as "unclear", but the statute made it', 'The rumor described the rule as "clear", but the statute made it', " clear", " unclear"),
        e('One blogger called the outcome "doubtful", but the official count made it', 'One blogger called the outcome "definite", but the official count made it', " definite", " doubtful"),
        e('The minutes quoted a "hesitant" observer, but the lead analyst sounded', 'The minutes quoted a "confident" observer, but the lead analyst sounded', " confident", " hesitant"),
        e('The article quoted the word "ambiguous", but the final diagram made it', 'The article quoted the word "obvious", but the final diagram made it', " obvious", " ambiguous"),
        e('The email repeated someone else saying "unsure", but the author was', 'The email repeated someone else saying "sure", but the author was', " certain", " uncertain"),
    ],
    "contrast": [
        e("At first the answer seemed uncertain, but after checking the details it was", "At first the answer seemed certain, but after checking the details it was", " certain", " uncertain"),
        e("The opening paragraph sounded unclear, but the appendix made the rule", "The opening paragraph sounded clear, but the appendix made the rule", " clear", " unclear"),
        e("The early estimate was doubtful, but the final count was", "The early estimate was definite, but the final count was", " definite", " doubtful"),
        e("The speaker began hesitant, but after the evidence she sounded", "The speaker began confident, but after the evidence she sounded", " confident", " hesitant"),
        e("The puzzle initially looked ambiguous, but after the hint it was", "The puzzle initially looked obvious, but after the contradiction it was", " obvious", " ambiguous"),
        e("The first draft made the claim unsure, but the revision made it", "The first draft made the claim sure, but the revision made it", " certain", " uncertain"),
    ],
    "format_shift": [
        e("Conclusion:\nEvidence status: verified\nConfidence level:", "Conclusion:\nEvidence status: unresolved\nConfidence level:", " certain", " uncertain"),
        e("Evidence status: all checks passed\nThe conclusion is", "Evidence status: checks conflict\nThe conclusion is", " clear", " unclear"),
        e("Case note\nFinal determination:", "Case note\nPending determination:", " definite", " doubtful"),
        e("Speaker profile\nAfter verification, the stance was", "Speaker profile\nBefore verification, the stance was", " confident", " hesitant"),
        e("Problem sheet\nWith the answer key, the next step is", "Problem sheet\nWithout the answer key, the next step is", " obvious", " ambiguous"),
        e("Audit log: reconciled\nResult:", "Audit log: unresolved discrepancy\nResult:", " certain", " uncertain"),
    ],
}

SENTIMENT_TRAIN: list[Entry] = [
    e("After the moving final scene, the movie felt", "After the clumsy final scene, the movie felt", " great", " terrible"),
    e("The attentive service at the small cafe was", "The careless service at the small cafe was", " good", " bad"),
    e("The new album's polished production sounded", "The new album's distorted production sounded", " excellent", " awful"),
    e("The mountain view from the room was", "The blocked view from the room was", " amazing", " poor"),
    e("The birthday dinner was", "The delayed birthday dinner was", " wonderful", " disappointing"),
    e("The repaired laptop worked out", "The repaired laptop failed again and felt", " great", " terrible"),
    e("The staff handled the complaint in a", "The staff ignored the complaint in a", " good", " bad"),
    e("The clean interface made the app", "The broken interface made the app", " excellent", " awful"),
    e("The concert encore was", "The concert sound mix was", " amazing", " poor"),
    e("The weekend trip was", "The canceled weekend trip was", " wonderful", " disappointing"),
    e("The dessert after dinner tasted", "The burnt dessert after dinner tasted", " great", " terrible"),
    e("The revised chapter was", "The rushed chapter was", " good", " bad"),
    e("The hotel upgrade was", "The hotel leak was", " excellent", " awful"),
    e("The guide's knowledge was", "The guide's preparation was", " amazing", " poor"),
    e("The quiet garden was", "The noisy garden was", " wonderful", " disappointing"),
    e("The team's comeback was", "The team's collapse was", " great", " terrible"),
    e("The replacement part was", "The cracked replacement part was", " good", " bad"),
    e("The translation quality was", "The corrupted translation quality was", " excellent", " awful"),
    e("The breakfast buffet was", "The stale breakfast buffet was", " amazing", " poor"),
    e("The museum tour was", "The shortened museum tour was", " wonderful", " disappointing"),
    e("The satisfying ending of the novel was", "The incoherent ending of the novel was", " great", " terrible"),
    e("The classroom explanation was", "The confused classroom explanation was", " good", " bad"),
    e("The theater seats were", "The broken theater seats were", " excellent", " awful"),
    e("The surprise performance was", "The missed performance was", " amazing", " poor"),
    e("The family reunion was", "The family argument was", " wonderful", " disappointing"),
    e("The software update was", "The buggy software update was", " great", " terrible"),
    e("The lunch special was", "The cold lunch special was", " good", " bad"),
    e("The customer support was", "The absent customer support was", " excellent", " awful"),
    e("The restored painting looked", "The damaged painting looked", " amazing", " poor"),
    e("The final gift was", "The missing final gift was", " wonderful", " disappointing"),
]

SENTIMENT_HELDOUT: list[Entry] = [
    e("The moving documentary was", "The misleading documentary was", " great", " terrible"),
    e("The new balanced headphones sounded", "The new distorted headphones sounded", " good", " bad"),
    e("The bakery's pastry was", "The bakery's spoiled pastry was", " excellent", " awful"),
    e("The fireworks show was", "The obstructed fireworks show was", " amazing", " poor"),
    e("The handwritten note was", "The forgotten note was", " wonderful", " disappointing"),
    e("The rescue plan worked out", "The rescue plan fell apart and felt", " great", " terrible"),
    e("The class discussion was", "The hostile class discussion was", " good", " bad"),
    e("The scholarship news was", "The rejection news was", " excellent", " awful"),
    e("The rooftop dinner was", "The soggy rooftop dinner was", " amazing", " poor"),
    e("The anniversary surprise was", "The ruined anniversary surprise was", " wonderful", " disappointing"),
    e("The orchestra performance was", "The out-of-tune orchestra performance was", " great", " terrible"),
    e("The neighborhood park looked", "The neglected neighborhood park looked", " good", " bad"),
    e("The final presentation was", "The chaotic final presentation was", " excellent", " awful"),
    e("The canyon hike was", "The muddy canyon hike was", " amazing", " poor"),
    e("The farewell party was", "The canceled farewell party was", " wonderful", " disappointing"),
    e("The new restaurant was", "The overhyped restaurant was", " great", " terrible"),
    e("The training session was", "The unprepared training session was", " good", " bad"),
    e("The repair service was", "The failed repair service was", " excellent", " awful"),
]

SENTIMENT_DECOUPLING: dict[str, list[Entry]] = {
    "lexical_reversal": [
        e('The review quoted the word "terrible", but my final judgment was', 'The blurb promised something "great", but the final product was', " great", " terrible"),
        e('The article mentioned a "bad" rumor, but the meal itself was', 'The article mentioned a "good" rumor, but the meal itself was', " good", " bad"),
        e('The comments expected it to be "awful", but the show was', 'The comments expected it to be "excellent", but the show was', " excellent", " awful"),
        e('Someone joked it would be "poor", but the view was', 'Someone joked it would be "amazing", but the view was', " amazing", " poor"),
        e('The preview warned of a "disappointing" night, but the event was', 'The preview promised a "wonderful" night, but the event was', " wonderful", " disappointing"),
        e('The headline used "terrible", but the actual review called it', 'The headline used "great", but the actual review called it', " great", " terrible"),
    ],
    "negation": [
        e("The movie was not terrible; it was", "The movie was not great; it was", " great", " terrible"),
        e("The meal was not bad; it was", "The meal was not good; it was", " good", " bad"),
        e("The performance was not awful; it was", "The performance was not excellent; it was", " excellent", " awful"),
        e("The view was not poor; it was", "The view was not amazing; it was", " amazing", " poor"),
        e("The party was not disappointing; it was", "The party was not wonderful; it was", " wonderful", " disappointing"),
        e("The update was not terrible anymore; it was", "The update was not great anymore; it was", " great", " terrible"),
    ],
    "quotation": [
        e('A critic called it "terrible", but I found it', 'A critic called it "great", but I found it', " great", " terrible"),
        e('The rumor labeled the cafe "bad", but we found it', 'The rumor labeled the cafe "good", but we found it', " good", " bad"),
        e('The poster quoted "awful", but the actual play was', 'The poster quoted "excellent", but the actual play was', " excellent", " awful"),
        e('The guidebook quoted a "poor" rating, but the trail was', 'The guidebook quoted an "amazing" rating, but the trail was', " amazing", " poor"),
        e('The message repeated "disappointing", but the visit was', 'The message repeated "wonderful", but the visit was', " wonderful", " disappointing"),
        e('The ad mocked the word "terrible", but the meal was', 'The ad repeated the word "great", but the meal was', " great", " terrible"),
    ],
    "contrast": [
        e("Although people said it was terrible, I thought it was", "Although the trailer looked great, the movie was", " great", " terrible"),
        e("Although the rumor said it was bad, dinner was", "Although the menu looked good, dinner was", " good", " bad"),
        e("Although early comments called it awful, the concert was", "Although early comments called it excellent, the concert was", " excellent", " awful"),
        e("Although the forecast looked poor, the hike was", "Although the photos looked amazing, the hike was", " amazing", " poor"),
        e("Although the email predicted disappointment, the evening was", "Although the invitation sounded wonderful, the evening was", " wonderful", " disappointing"),
        e("Although the first act seemed terrible, the full show was", "Although the first act seemed great, the full show was", " great", " terrible"),
    ],
    "format_shift": [
        e("Review note: audience applauded\nFinal judgment:", "Review note: audience walked out\nFinal judgment:", " great", " terrible"),
        e("Meal log: fresh ingredients\nOverall quality:", "Meal log: stale ingredients\nOverall quality:", " good", " bad"),
        e("Performance scorecard: precise and moving\nFinal rating:", "Performance scorecard: sloppy and painful\nFinal rating:", " excellent", " awful"),
        e("Travel entry: clear sky over the canyon\nView quality:", "Travel entry: fog blocked the canyon\nView quality:", " amazing", " poor"),
        e("Event recap: everyone left smiling\nFinal feeling:", "Event recap: everyone left frustrated\nFinal feeling:", " wonderful", " disappointing"),
        e("Product note: fixed defects\nCustomer reaction:", "Product note: defects returned\nCustomer reaction:", " great", " terrible"),
    ],
}


def records_from_entries(
    construct: str,
    split: str,
    axis: str,
    entries: list[Entry],
    class_a_label: str,
    class_b_label: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index, (a_prompt, b_prompt, a_target, b_target, notes) in enumerate(entries, start=1):
        pair_id = f"{construct}_{split}_{axis}_{index:03d}"
        for role, prompt, label in (
            ("class_a", a_prompt, "class_a"),
            ("class_b", b_prompt, "class_b"),
        ):
            records.append(
                {
                    "id": f"{pair_id}_{role}",
                    "construct": construct,
                    "split": split,
                    "decoupling_axis": axis,
                    "prompt": prompt,
                    "class_a_label": class_a_label,
                    "class_b_label": class_b_label,
                    "class_a_target": a_target,
                    "class_b_target": b_target,
                    "label": label,
                    "template_id": f"{construct}_{axis}_{index:02d}_{role}",
                    "notes": notes,
                    "pair_id": pair_id,
                    "pair_role": role,
                }
            )
    return records


def filter_invalid_entries(
    entries: list[Entry],
    valid_targets: set[str],
) -> list[Entry]:
    return [entry for entry in entries if entry[2] in valid_targets and entry[3] in valid_targets]


def validate_size(construct: str, train: list[dict[str, str]], heldout: list[dict[str, str]], decoupling: list[dict[str, str]]) -> None:
    if not 50 <= len(train) <= 100:
        raise ValueError(f"{construct} train size is out of target range: {len(train)}")
    if not 30 <= len(heldout) <= 60:
        raise ValueError(f"{construct} heldout size is out of target range: {len(heldout)}")
    counts = Counter(record["decoupling_axis"] for record in decoupling)
    for axis in DECOUPLING_AXES:
        if not 10 <= counts[axis] <= 25:
            raise ValueError(f"{construct} {axis} size is out of target range: {counts[axis]}")


def build_construct(
    construct: str,
    train_entries: list[Entry],
    heldout_entries: list[Entry],
    decoupling_entries: dict[str, list[Entry]],
    class_a_label: str,
    class_b_label: str,
    valid_targets: set[str],
    output_root: Path,
) -> dict[str, list[dict[str, str]]]:
    train_entries = filter_invalid_entries(train_entries, valid_targets)
    heldout_entries = filter_invalid_entries(heldout_entries, valid_targets)
    decoupling_entries = {
        axis: filter_invalid_entries(entries, valid_targets)
        for axis, entries in decoupling_entries.items()
    }
    train = records_from_entries(construct, "train", "ordinary", train_entries, class_a_label, class_b_label)
    heldout = records_from_entries(construct, "heldout", "ordinary", heldout_entries, class_a_label, class_b_label)
    decoupling: list[dict[str, str]] = []
    for axis in DECOUPLING_AXES:
        decoupling.extend(
            records_from_entries(
                construct,
                "decoupling",
                axis,
                decoupling_entries[axis],
                class_a_label,
                class_b_label,
            )
        )
    validate_size(construct, train, heldout, decoupling)
    for split, records in (("train", train), ("heldout", heldout), ("decoupling", decoupling)):
        write_jsonl(dataset_file(construct, split, output_root), records)
    return {"train": train, "heldout": heldout, "decoupling": decoupling}


def write_notes(all_records: dict[str, dict[str, list[dict[str, str]]]], output_root: Path) -> None:
    notes_path = artifact_path(output_root) / "datasets" / "dataset_notes.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Quality Notes",
        "",
        "The dataset is intentionally small and template-curated. Each example was written to make the target construct explicit while allowing the decoupling axis to stress a different validity target.",
        "",
        "Sentiment is included as a familiar sanity-check construct. Certainty/uncertainty is the primary construct because the intended contribution is the controlled construct-validity matrix, not sentiment analysis.",
        "",
        "All class targets used by the generated JSONL files were validated as single GPT-2 tokens with leading spaces. Entries whose target pair failed single-token validation are excluded before size checks.",
        "",
        "Known caveat: several format-shift examples intentionally share a surface prompt across class roles and rely on the target-token contrast. They are useful for measuring target-token behavior but should not be overread as naturalistic semantic minimal pairs.",
        "",
        "Counts:",
    ]
    for construct, splits in all_records.items():
        lines.append(f"- {construct}:")
        for split, records in splits.items():
            counts = Counter(record["decoupling_axis"] for record in records)
            axis_counts = ", ".join(f"{axis}={counts[axis]}" for axis in ALL_AXES if counts[axis])
            lines.append(f"  - {split}: n={len(records)} ({axis_counts})")
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {notes_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    all_targets = {
        entry[2]
        for entries in (
            CERTAINTY_TRAIN,
            CERTAINTY_HELDOUT,
            SENTIMENT_TRAIN,
            SENTIMENT_HELDOUT,
            *CERTAINTY_DECOUPLING.values(),
            *SENTIMENT_DECOUPLING.values(),
        )
        for entry in entries
    } | {
        entry[3]
        for entries in (
            CERTAINTY_TRAIN,
            CERTAINTY_HELDOUT,
            SENTIMENT_TRAIN,
            SENTIMENT_HELDOUT,
            *CERTAINTY_DECOUPLING.values(),
            *SENTIMENT_DECOUPLING.values(),
        )
        for entry in entries
    }
    valid_target_ids = validate_single_token_targets(
        all_targets,
        encode_text=lambda text: encode_text(model, text),
        decode_tokens=lambda token_ids: decode_tokens(model, token_ids),
    )
    valid_targets = set(valid_target_ids)

    all_records = {
        "certainty": build_construct(
            "certainty",
            CERTAINTY_TRAIN,
            CERTAINTY_HELDOUT,
            CERTAINTY_DECOUPLING,
            "certain",
            "uncertain",
            valid_targets,
            args.output_root,
        ),
        "sentiment": build_construct(
            "sentiment",
            SENTIMENT_TRAIN,
            SENTIMENT_HELDOUT,
            SENTIMENT_DECOUPLING,
            "positive",
            "negative",
            valid_targets,
            args.output_root,
        ),
    }
    write_notes(all_records, args.output_root)
    print("Wrote processed datasets")


if __name__ == "__main__":
    main()
