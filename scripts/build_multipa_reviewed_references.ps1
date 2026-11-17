param(
    [string]$InputCsv = "MultiPA\multipa_reference_texts.csv",
    [string]$ReportRoot = "runs\capt_multipa_modeA_reference_text_pcc\reports",
    [string]$OutputCsv = "MultiPA\multipa_reference_texts_reviewed.csv"
)

$ErrorActionPreference = "Stop"

# These edits are limited to cases where the two ASR outputs and the utterance
# context expose a clear lexical error. Ambiguous morphology, fillers, and
# non-native grammar are deliberately preserved for audio review.
$corrections = @{
    "fc73fa8c-ba24-455f-8089-25b039544b56---ba1d3b80-847d-47fb-aa10-497635abed4.wav" = @{
        text = "Yes, I am capable of fixing simple things such as a leaky faucet or a blown fuse. However, for more complex issues, I will seek the help of a professional to ensure the problem is properly resolved."
        note = "Removed the contextually spurious word 'false' from 'blown fuse'; WhisperX large-v3 agrees."
    }
    "f353a88e-cfcb-44d7-a23b-f6c26c9ea908---a5d60c2b-f9fe-4e81-9e83-0c9e7678254.wav" = @{
        text = "Hmm. I don't quite know about Da Vinci, but I do know about Michelangelo. Do you know him?"
        note = "Restored the name 'Da Vinci'; topic context and WhisperX large-v3 agree."
    }
    "f1ab2663-fe7a-48d0-b504-043a718d8906---71e838e8-b508-4f2a-aa33-42a49b78eff.wav" = @{
        text = "Well, every afternoon I would eat snacks just like a little chocolate cake or chocolate cookies with a simple coffee like Americano."
        note = "Corrected the beverage name 'American Chino' to 'Americano'; WhisperX large-v3 agrees."
    }
    "bcee2ca5-8483-408a-93e3-50528f6da939---5fbdf9ec-5c7d-4536-b58f-012c983929f.wav" = @{
        text = "For species animals, I think it needs to set up a natural reserve that can be good for them and protect them well."
        note = "Resolved the shared ASR confusion 'natural device/service' as 'natural reserve' from the protection context."
    }
    "2090115f-ca26-420e-81c3-28ad672f8c42---e6280ff5-183a-4087-8e69-8ee96c6977a.wav" = @{
        text = "No, I don't like singing, because when I sing a song, my parents and my friends will tell me it's terrible. It's not good."
        note = "Corrected 'seeing/see' to 'singing/sing' from the repeated song context."
    }
    "516be824-f565-43b1-ba43-98bbfd69109f---2b1f4d68-3935-4960-b815-cc0815e676b.wav" = @{
        text = "Yes, it was really insightful. I learned a lot about how to create sustainable projects that benefit my community."
        note = "Resolved 'situational/situatable projects' as the contextually standard phrase 'sustainable projects'."
    }
    "162bf8af-521d-434b-9f93-89e16add007a---a446c11c-8aad-4769-a2bd-a73905ab726.wav" = @{
        text = "Yes, I can learn a lot of music per day, like 20 or 30 songs, and most of them are the same type of music."
        note = "Corrected 'same time' to 'same type of music'; WhisperX large-v3 and sentence context agree."
    }
    "7b34d27c-a0dc-47ae-a8bd-956f02995342---b854e45b-358b-491a-b015-c2d7cc207dc.wav" = @{
        text = "Well, I told my foreign friends about this city, and I guess they may like the unique Chinese culture. If one day they really come, I think I will treat them well."
        note = "Resolved 'foreigner friends/turn them wheel' as 'foreign friends/treat them well' from context and ASR agreement."
    }
    "7b34d27c-a0dc-47ae-a8bd-956f02995342---70f4dd53-bae8-40aa-9ade-9296db1707a.wav" = @{
        text = "I think visiting the zoo is very meaningful. We can see many animals and learn a lot about them. For example, you can see pandas eating bamboo and tigers eating meat. It's very useful, I think."
        note = "Resolved the corrupted 'Gram-C' phrase using the repeated zoo/animal context and WhisperX large-v3."
    }
    "7b34d27c-a0dc-47ae-a8bd-956f02995342---64a6c36d-62db-4123-acf8-f882c18d1c4.wav" = @{
        text = "There are two situations where I will be very focused. Firstly, when the deadline is approaching, I will focus on my work. Secondly, when I dance, I am very focused on listening to music."
        note = "Resolved 'guidelines/dialogue approachings' as 'deadline is approaching' from both ASR hypotheses and context."
    }
    "1abdb5b5-298d-49c1-ab9f-29f6f11e0a9d---b481a3b7-5d1c-4518-b64e-7193dfded9a.wav" = @{
        text = "I think visiting the zoo is very meaningful. We can see many animals, and there are a lot of wild animals, just like pandas eating bamboo and tigers eating meat. It's very interesting, I think."
        note = "Corrected the corrupted 'fellas' segment using WhisperX large-v3 and the surrounding animal examples."
    }
}

function Normalize-Words([string]$Text) {
    return ([regex]::Replace($Text.ToLowerInvariant(), "[^a-z0-9']+", " ")).Trim()
}

function Get-WhisperXText([string]$Audio) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($Audio)
    $path = Join-Path $ReportRoot ("multipa_modeA_" + $stem + "\report.json")
    if (-not (Test-Path -LiteralPath $path)) {
        return ""
    }
    $json = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))
    $match = [regex]::Match(
        $json,
        '(?s)"asr"\s*:\s*\{\s*"text"\s*:\s*"((?:\\.|[^"\\])*)"'
    )
    if (-not $match.Success) {
        return ""
    }
    return [regex]::Unescape($match.Groups[1].Value)
}

$reviewed = foreach ($row in (Import-Csv -LiteralPath $InputCsv)) {
    $original = [string]$row.reference_text
    $whisperx = Get-WhisperXText $row.audio
    $correction = $corrections[$row.audio]
    if ($null -ne $correction) {
        $referenceText = $correction.text
        $status = "dual_asr_context_corrected"
        $notes = $correction.note
    }
    elseif ($whisperx -and (Normalize-Words $original) -eq (Normalize-Words $whisperx)) {
        $referenceText = $original
        $status = "dual_asr_word_agreement"
        $notes = "The two ASR transcripts agree at normalized word level; punctuation may differ."
    }
    else {
        $referenceText = $original
        $status = "pending_audio_review"
        $notes = "ASR hypotheses differ lexically; retained the original transcript pending listening-based review."
    }

    [pscustomobject][ordered]@{
        audio = $row.audio
        wav_path = $row.wav_path
        reference_text = $referenceText
        reference_text_original = $original
        whisperx_large_v3_text = $whisperx
        transcription_model = $row.transcription_model
        transcription_review_status = $status
        transcription_review_notes = $notes
        language_probability = $row.language_probability
        num_annotators = $row.num_annotators
        mean_accuracy = $row.mean_accuracy
        mean_fluency = $row.mean_fluency
        mean_prosody = $row.mean_prosody
        accuracy_scores = $row.accuracy_scores
        fluency_scores = $row.fluency_scores
        prosody_scores = $row.prosody_scores
    }
}

$reviewed | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8

$reviewed | Group-Object transcription_review_status | Sort-Object Name | ForEach-Object {
    Write-Host ("{0}: {1}" -f $_.Name, $_.Count)
}
Write-Host "Wrote $($reviewed.Count) rows to $OutputCsv"
