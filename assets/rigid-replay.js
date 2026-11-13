// Display a body track using the room's own geometry/materials. Geometry is
// already baked in the room's bind pose; sample body-relative displacement.
import {AnimationClip, Group, Matrix4, Mesh, QuaternionKeyframeTrack, VectorKeyframeTrack} from 'three';

export function buildRigidReplay(motion,room,taskId) {
  const valid=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
  if(motion.schema!=='partner-rigid-motion-v1'||motion.decodedModelSha256!==room.userData.decodedModelSha256||
     motion.taskId!==taskId||!Array.isArray(motion.tracks)||!motion.tracks.length)throw Error('Motion does not match this room and task.');
  const times=motion.times;
  if(!Array.isArray(times)||times.length<2||times[0]!==0||times.some((t,i)=>!Number.isFinite(t)||(i&&t<=times[i-1])))throw Error('Invalid replay timestamps.');
  const nodes=new Map(),hidden=new Set(),scene=new Group(),keys=[];
  room.traverse(o=>{if(Number.isInteger(o.userData.replayNodeIndex))nodes.set(o.userData.replayNodeIndex,o);});
  for(const [i,track] of motion.tracks.entries()){
    if(!valid(track.pivot,3)||!track.nodes?.length||!track.positions?.every(p=>valid(p,3))||!track.rotations?.every(q=>valid(q,4))||
       track.positions.length!==times.length||track.rotations.length!==times.length)throw Error('Incomplete recorded motion.');
    const group=new Group();group.name='recorded_body_'+i;scene.add(group);
    const invPivot=new Matrix4().makeTranslation(...track.pivot.map(a=>-a));
    for(const n of track.nodes){
      const source=nodes.get(n);if(!source)throw Error('Missing recorded room part: '+n);
      let count=0;
      source.traverse(o=>{
        if(!o.isMesh)return;
        if(hidden.has(o)||!o.userData.replayBindWorld)throw Error('Overlapping or unbound replay geometry.');
        hidden.add(o);count++;
        if(!o.visible)return;
        const copy=new Mesh(o.geometry,o.material);copy.matrixAutoUpdate=false;
        copy.matrix.multiplyMatrices(invPivot,new Matrix4().fromArray(o.userData.replayBindWorld));group.add(copy);
      });
      if(!count)throw Error('Recorded part has no geometry.');
    }
    keys.push(new VectorKeyframeTrack(group.name+'.position',times,track.positions.flat()),new QuaternionKeyframeTrack(group.name+'.quaternion',times,track.rotations.flat()));
  }
  if(motion.hideObjectIds?.length){
    const ids=new Set(motion.hideObjectIds), matched=new Set();
    room.traverse(o=>{if(o.isMesh&&ids.has(o.userData.objectId)){matched.add(o.userData.objectId);hidden.add(o);}});
    if(matched.size!==ids.size)throw Error('The mechanism does not match the room display.');
  }
  if(!scene.children.some(g=>g.children.length))throw Error('The recorded geometry is hidden.');
  return {scene,animations:[new AnimationClip('Recorded task',times[times.length-1],keys)],hidden};
}
