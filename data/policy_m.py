"""Exact historical enrollment eligibility predicate."""
def legal_enrollment(t,e):
 return (t['dataset']==e['dataset'] and e['split']=='train' and t['singer_id']==e['singer_id'] and t['song_id']!=e['song_id'] and t['source_recording_group']!=e['source_recording_group'] and t['source_audio_sha256']!=e['source_audio_sha256'])
