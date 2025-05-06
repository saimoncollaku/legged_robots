NOTE: I was able to do the TO for every request motion.

# Ex.9
Disclaimer is that the pd gains for thigh and knee joints are different.
I noticed that knee generally required faster tracking / forces. 
The pd gains for each motion can be found in "pd_ff_gains.txt".

For squat:
The performance and tracking is nearly perfect.

For walk and backflip180:
The tracking is pretty good, but in some instances the control is slow, 
in the walk there are some leg raises that seem kind of drunk and, in the landing 
for the backflip180, the hind legs finish their movement late.

For trot, jump and bound:
Because the movement of these motion required high speed, the tracking really 
cannot really keep up, especially for the jump. The movements resemble the TO poses,
so the controller is at least pretty stable.


For backflip:
Controller cannot keep-up at all. Bad tracking and performance.


# Ex.10
Disclaimer, I added a gain on the feedforward term.
The ff gains for each motion can be found in "pd_ff_gains.txt".

For any choice of gains, there was no hope. The ff controller is unstable.
Basically the joints keep accelerating for every motion.
Maybe the reasons are (from the github issue):
1 there are joints range limitations declared on the mujoco file, when robot exceeds
those ranges, a counteracting force gets applied
2 because we are upsampling the trajectory we are implicitly assuming that the 
SRB dynamic equations for the TO are linear, they are not, so we get "bad" usable feet
forces

# Ex.11
Honestly even by adding the FF term, there wasn't that much improvement. 

Nothing happened in the squatting motion. The tracking was already perfect.

Also jump did not improve that much, the robot is still not able to reach the height 
of the TO.

For walk, trot and bound: there are some improvements. For example walk has better tracking
during the leg raises, basically the robot seems "less drunk". Also in the trot and bound, the robot 
is able to move further, due to the faster tracking.

Interestingly, for the backflip180, for any choice of ff gains, the tracking would get worse than the 
one with just PD. Super interestingly, with ff gains of 5, I could get the robot to do a DOUBLE backflip.
Shown at the time 51 - 56 of the video. 

Also for the backflip, no ff choice of gains could get fast tracking. So, I cheated: to the controller I added 
this term ->  - k_a * q_dotdot_desired 
Basically, it's a term dependend on the desired acceleration from the TO, k_a is the gain.
Also I went even deeper with the gains cause I noticed this:
1 for the hind legs it's not really important if we overshoot, cause that would just result in a height reached
which is not really a problem imo
2 for the front legs, the tracking is important cause if it's not the hind leg push would happend at the wrong position.
Due to this I added, I divide the gains the for the hind (which are much higher), with the gains for the front 
(lower gains to avoid overshoot & ringing). The effect are still not the same of the TO, but at least the robot did the backflip.
Gains can be found in "backflip_gains.txt". The backflip is shown at time 56 - 1.16 at various angles.