"""Exact historical enrollment eligibility predicate."""
def legal_enrollment(target, candidate):
    return (candidate['split'] == 'train'
            and target['singer_id'] == candidate['singer_id']
            and target['song_id'] != candidate['song_id']
            and target['source_recording_group'] != candidate['source_recording_group']
            and target['source_audio_sha256'] != candidate['source_audio_sha256'])
