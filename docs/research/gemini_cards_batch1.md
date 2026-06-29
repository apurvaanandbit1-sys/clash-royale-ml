Clash Royale Machine Learning Knowledge Base: Card Attribute Analysis
Methodological Framework and Classification Standards
The following research report provides a rigorously structured, evidence-based data schema designed to serve as the single source of truth for a Clash Royale machine learning pipeline. In strict adherence to the directive prioritizing absolute accuracy over unverified completeness, this analysis relies exclusively on cross-referenced, high-quality data from official documentation, RoyaleAPI, and competitive strategy analysis.

To ensure the machine learning model ingests standardized parameters, all combat attributes have been evaluated at the Tournament Standard (Level 11). Whenever empirical data for a specific attribute is absent from the research corpus, or when subjective community consensus cannot be quantified objectively, the attribute is explicitly flagged as "Requires Human Decision" to prevent the introduction of hallucinated variables into the pipeline.

The threshold parameters for combat categorizations have been algorithmically defined utilizing the provided dataset:   

HP Level (Level 11): VERY_LOW (<200), LOW (200–800), MEDIUM (801–1500), HIGH (1501–2500), VERY_HIGH (>2500).

DPS Level (Level 11): VERY_LOW (<50), LOW (50–100), MEDIUM (101–200), HIGH (201–300), VERY_HIGH (>300).

Attack Speed: VERY_FAST (<=0.8s), FAST (0.9s–1.2s), MEDIUM (1.3s–1.5s), SLOW (1.6s–2.5s), VERY_SLOW (>2.5s).

Attack Range: MELEE (Internally categorized as Short <=0.8, Medium 1.2, Long 1.6), SHORT (2.0–3.5), MEDIUM (4.0–5.0), LONG (5.5–6.5), VERY_LONG (>=7.0).

Card Profiles
Knight
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Medium
Attack Style	Melee
Targets	Ground
Combat	HP Level	HIGH
DPS Level	MEDIUM
Attack Speed	FAST
Attack Range	MELEE (Medium)
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	True
Unique Abilities	Hero Form (Triumphant Taunt)
Strategic	Primary Role	Mini-Tank
Secondary Roles	Defensive Support
Win Condition	False
Support Card	True
Cycle Card	False
Tags	Generated Tags	troop, ground, melee, single_target, mini_tank, defense
The structural profile of the Knight designates him fundamentally as a ground-based troop characterized by a medium movement velocity. Operating exclusively within the physical domain, his attack style is a single-target melee strike that is fundamentally incapable of targeting aerial units. In terms of combat metrics at the standard Level 11 benchmark, his 1,766 hitpoints secure a highly durable classification, while his damage output of 168 per second places his DPS in the medium tier. His attack speed of 1.2 seconds dictates a fast classification, and his official strike reach of 1.2 tiles strictly categorizes his range as a medium melee engagement. The Knight lacks any innate splash damage, charge mechanics, knockback, or electrical reset capabilities.   

While the base card possesses no innate death or spawn abilities, it features a highly impactful Evolution Ability that grants a reflective shield, mathematically minimizing incoming damage by 60% while the unit is moving and not actively striking. Furthermore, the Knight possesses a Hero Form that incorporates the "Triumphant Taunt" mechanic, forcing enemy retargeting within a 6.5-tile radius. Strategically, the Knight serves primarily as a mini-tank, routinely deployed to absorb damage and distract high-value targets, fulfilling a secondary role as defensive support. He is unequivocally not a win condition, as he targets troops and cannot reliably bypass defenses to secure tower destruction, but he excels as a supportive anchor for other win conditions like the Goblin Barrel. Consequently, the assigned tags reflect his ground-based, single-target, mini-tank utility.   

Valkyrie
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Medium
Attack Style	Splash
Targets	Ground
Combat	HP Level	HIGH
DPS Level	MEDIUM
Attack Speed	MEDIUM
Attack Range	MELEE (Medium)
Splash Damage	True
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	True
Unique Abilities	None
Strategic	Primary Role	Splash Defender / Mini-Tank
Secondary Roles	Counter-pusher
Win Condition	False
Support Card	True
Cycle Card	False
Tags	Generated Tags	troop, ground, melee, splash, mini_tank, control, defense
The Valkyrie is categorized as a deployable troop moving at a standard medium pace. Her defining combat characteristic is a 360-degree splash attack that impacts all adjacent ground units simultaneously, completely ignoring flying entities. At Level 11, the Valkyrie possesses 1,907 hitpoints, landing her securely inside the high durability bracket, while her damage per second rating of 177 places her in the medium tier. She executes an attack every 1.5 seconds. Her physical reach is uniquely coded; while her splash radius extends 2.4 tiles from her center, the trigger range to initiate the attack is exactly 1.2 tiles, defining her as a medium-ranged melee troop. She possesses no charge, knockback, or stun capabilities in her base form.   

The Valkyrie does not possess death damage, spawn effects, healing, freezing, or chaining capabilities. She does, however, feature an Evolution Ability. When deployed in her evolved state, her attacks generate a localized Tornado effect that pulls enemies within a 5.5-tile radius toward her, inflicting minor supplementary chip damage and acting as an advanced form of crowd control. Strategically, she is widely utilized as a premier splash defender and mini-tank, routinely dropped directly onto enemy swarms or supporting backline troops (such as Witches or Musketeers) to instantly neutralize them while absorbing heavy punishment. She frequently transitions into a secondary role as a counter-pusher, tanking for win conditions like the Hog Rider immediately after successfully defending. She is not a win condition herself, as she is too easily distracted by enemy placements to reliably secure tower destruction. Her generated tags emphasize her role as a splash-oriented, defensive mini-tank.   

Firecracker
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Fast
Attack Style	Ranged Splash (Piercing)
Targets	Air & Ground
Combat	HP Level	LOW
DPS Level	MEDIUM
Attack Speed	VERY_SLOW
Attack Range	LONG
Splash Damage	True
Can Attack Air	True
Can Attack Ground	True
Charge Mechanic	False
Knockback	True (Self-Recoil)
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	True
Unique Abilities	None
Strategic	Primary Role	Ranged Support / Air Defense
Secondary Roles	Spell Bait
Win Condition	False
Support Card	True
Cycle Card	False
Tags	Generated Tags	troop, ranged, splash, support, air_defense
Structurally, the Firecracker is a fast-moving troop that utilizes a highly specific piercing splash attack, firing projectiles that burst into multiple shrapnel pieces capable of damaging both air and ground targets located behind the initial point of impact. Her combat statistics at Level 11 reveal extreme fragility, with only 304 hitpoints placing her in the low durability threshold. However, the overlapping nature of her shrapnel generates 105 DPS, pushing her into the medium damage tier. She is hampered by a very slow attack speed, requiring 3.0 seconds between shots, but benefits from a long engagement distance of 6.0 tiles. Notably, the Firecracker features a unique knockback mechanic applied solely to herself; she experiences a 1.5-tile self-recoil every time she discharges her weapon, allowing her to dodge slower melee attacks.   

The unit lacks traditional deployment or death abilities, but her Evolution Ability introduces a lingering spark mechanic that deals sustained area damage over time to any targets caught within the residual blast zones of her shrapnel. Strategically, the Firecracker acts as a potent ranged support and air defense unit, providing significant area denial from behind allied tanks. A critical secondary role is her function as "spell bait"; due to her low health pool and high threat level, she frequently forces opponents to expend spells like Arrows, thereby protecting other fragile units within the player's deck. The Firecracker is an easily distracted support troop and does not target towers primarily, disqualifying her from the win condition classification. Her generated tags isolate her identity as a ranged, splash-damage support unit.   

Giant
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Slow
Attack Style	Melee
Targets	Buildings
Combat	HP Level	VERY_HIGH
DPS Level	MEDIUM
Attack Speed	MEDIUM
Attack Range	MELEE (Long)
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	False
Unique Abilities	Hero Form
Strategic	Primary Role	Win Condition
Secondary Roles	Tank
Win Condition	True
Support Card	False
Cycle Card	False
Tags	Generated Tags	troop, ground, melee, tank, win_condition, building_target
The Giant is a foundational troop defined by a slow movement speed and a strict adherence to targeting only buildings and Crown Towers, entirely ignoring enemy troops. He engages these structures through single-target melee strikes. At Level 11, the Giant boasts an enormous 4,090 hitpoints, earning him a very high classification for durability. He delivers 168 damage per second, equating to a medium DPS level, with an attack triggering every 1.5 seconds. The Giant's physical reach was officially adjusted to 1.6 tiles, classifying his engagement distance precisely as long melee. He possesses no splash damage, charge capabilities, knockback force, or reset utility, nor can he interact with flying entities.   

The Giant lacks death damage, spawning mechanics, freezing, or raging auras. Furthermore, there is no evolutionary variant present in the current dataset. He does, however, possess a Hero Form containing a unique ability that temporarily alters his targeting paradigm, allowing him to launch a projectile capable of damaging enemy troops. From a strategic standpoint, the Giant is universally acknowledged as a primary win condition. Because he exclusively targets buildings and absorbs massive amounts of damage, his entire architectural purpose is to act as the spearhead for "Beatdown" archetypes, reliably generating tower damage while shielding fragile, high-damage support troops positioned behind him. He is inherently not a cycle or support card due to his 5-Elixir cost and specific targeting behavior. His tags isolate him as a primary ground tank and structural win condition.   

Miner
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Fast (Burrowing)
Attack Style	Melee
Targets	Ground
Combat	HP Level	Requires Human Decision
DPS Level	Requires Human Decision
Attack Speed	Requires Human Decision
Attack Range	MELEE (Medium)
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	False
Unique Abilities	Unrestricted Deployment, Reduced Tower Damage
Strategic	Primary Role	Hybrid Win Condition
Secondary Roles	Chip Damage / Assassin
Win Condition	True
Support Card	True
Cycle Card	False
Tags	Generated Tags	troop, ground, melee, mini_tank, win_condition, unrestricted_placement
The Miner is a unique troop characterized by an unrestricted burrowing movement mechanism that allows him to bypass the traditional river boundary, popping up instantly anywhere on the battlefield. Once deployed, he executes single-target melee strikes exclusively against ground targets. Precise Level 11 statistics for hitpoints, DPS, and attack speed are absent from the provided research corpus, necessitating a "Requires Human Decision" flag to prevent automated hallucination, though textually he is described as possessing moderate health. His attack range is explicitly defined as 1.2 tiles, landing in the medium melee classification. He lacks splash damage, charge mechanics, knockback, or stun abilities.   

The Miner features no death or spawn damage, nor does he spawn secondary units or possess an Evolution. His defining unique ability is his unrestricted placement, which is balanced by a secondary mechanical constraint: his damage is algorithmically reduced to only 25% of its base value when striking Crown Towers. Strategically, the Miner's classification presents a significant disagreement across trustworthy sources. Analytical databases like RoyaleAPI firmly categorize him as a primary win condition due to his ability to guarantee tower connections. Conversely, high-level competitive analysis argues that his heavily reduced tower damage makes him a "secondary win condition" or "chip damage" generator, forcing him to act primarily as an assassin for backline troops (e.g., Magic Archer) or as a mini-tank supporting true win conditions like Wall Breakers or Balloon. Consequently, he is classified herein as a Hybrid Win Condition that heavily overlaps with support and assassination roles.   

Mini P.E.K.K.A.
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Fast
Attack Style	Melee
Targets	Ground
Combat	HP Level	MEDIUM
DPS Level	VERY_HIGH
Attack Speed	SLOW
Attack Range	MELEE (Medium)
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	False
Unique Abilities	Hero Form
Strategic	Primary Role	Tank Killer / Defensive Pivot
Secondary Roles	Mini-Tank
Win Condition	False
Support Card	True
Cycle Card	False
Tags	Generated Tags	troop, ground, melee, tank_killer, high_damage, defense, single_target
Structurally, the Mini P.E.K.K.A. is a rapidly moving troop that must close distance to utilize a massive sword for single-target melee strikes against ground entities. At Level 11, the unit commands 1,433 hitpoints, placing it in the medium durability bracket. Its defining combat trait is an immense damage output; generating 471 damage per second at Level 11, it secures a very high DPS classification, prioritized for heavy single-target destruction. This high DPS is achieved despite a slow attack interval of 1.6 seconds. The Mini P.E.K.K.A. operates within a medium melee range of 1.2 tiles and possesses no splash damage, charge mechanics, knockback, or stun capabilities.   

The card contains no death damage, freezing, chaining, or evolutionary mechanics. However, it can be upgraded to a Hero variant within specific modes, altering its baseline interactions. From a strategic perspective, the Mini P.E.K.K.A. is the premier "Tank Killer." It is deployed primarily on defense to instantly delete high-health threats like the Giant, Golem, or Hog Rider before they can severely damage Crown Towers. Once the defensive sequence concludes, its 1,433 HP allows it to act as a formidable mini-tank during a counter-push. Despite dealing catastrophic damage if it reaches a Crown Tower, it is strictly classified as a support card rather than a win condition. Because it targets troops, it is exceptionally easy to distract with cheap swarms (e.g., Skeletons) and cannot be relied upon to consistently bypass defenses. Its tags emphasize its status as a high-damage defensive specialist.   

Skeletons
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Fast
Attack Style	Melee
Targets	Ground
Combat	HP Level	VERY_LOW
DPS Level	LOW
Attack Speed	FAST
Attack Range	MELEE (Short)
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	True
Unique Abilities	None
Strategic	Primary Role	Distraction / Cycle
Secondary Roles	Cheap DPS
Win Condition	False
Support Card	False
Cycle Card	True
Tags	Generated Tags	troop, ground, melee, swarm, cycle, distraction
The Skeletons card deploys multiple distinct, fast-moving units that strike with short swords, limiting their engagement exclusively to ground targets in close quarters. At Level 11, an individual skeleton contains exactly 81 hitpoints, rendering it vulnerable to almost any instance of damage and cementing its very low durability rating. Each skeleton generates 73 DPS; while low individually, three skeletons left ignored can output highly efficient aggregate damage against distracted tanks. They strike rapidly every 1.1 seconds and require point-blank proximity of 0.5 tiles (short melee) to engage. They possess no splash, charge, or knockback properties.   

In their base form, they do not possess any death, spawn, or healing abilities. However, they feature a highly unique Evolution Ability. Evolved Skeletons possess a multiplying mechanic, allowing them to spontaneously spawn additional skeletons (up to a predetermined maximum cap) every time an existing skeleton successfully lands a melee strike. Strategically, the Skeletons fulfill a vital cycle and distraction role. Costing only 1 Elixir—the cheapest tier available—they are frequently utilized to artificially "cycle" back to a win condition, or to pull high-threat, single-target enemies (like Mini P.E.K.K.A. or Prince) into the center of the arena, forcing them to waste time and attacks. They are entirely incapable of reaching a tower reliably, definitively ruling out any classification as a win condition. The generated tags highlight their function as a cheap, swarming distraction.   

P.E.K.K.A.
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Slow
Attack Style	Melee
Targets	Ground
Combat	HP Level	Requires Human Decision
DPS Level	Requires Human Decision
Attack Speed	SLOW
Attack Range	MELEE (Long)
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	True
Unique Abilities	None
Strategic	Primary Role	Defensive Tank / Tank Killer
Secondary Roles	Counter-pusher
Win Condition	False (Disputed)
Support Card	False
Cycle Card	False
Tags	Generated Tags	troop, ground, melee, tank_killer, high_damage, defense, heavy
The P.E.K.K.A. is a heavy troop characterized by lumbering, slow movement and the utilization of a massive sword for devastating, single-target ground strikes. Exact Level 11 statistics for her hitpoints and damage per second are absent from the provided research corpus, necessitating a "Requires Human Decision" flag, though textual evidence repeatedly confirms she possesses "extremely high hitpoints" and packs a "huge punch" capable of instantly destroying support units. She is plagued by a notably slow attack animation but benefits from a long melee reach of 1.6 tiles. She is entirely single-target, rendering her highly vulnerable to swarms, and cannot attack aerial units.   

She does not inherently possess death damage, spawn units, or freezing abilities. Her Evolution Ability, however, introduces a remarkable "Butter-Heal" skill, allowing her to heal a portion of her health every time she lands a killing blow on an enemy troop. This mechanic theoretically allows her hitpoints to exceed her standard maximum deployment health. Strategically, the 7-Elixir P.E.K.K.A. is deployed almost exclusively as an impenetrable defensive wall to obliterate opposing win conditions (such as the Golem or Mega Knight) before translating that surviving health into a massive counter-push. Her classification as a win condition is a source of intense community dispute. While she can single-handedly destroy a tower if she connects, RoyaleAPI and analytical experts classify her as a "Key Card" or "Pseudo Win-Condition" because she targets troops and is far too easily distracted by 1-elixir skeletons to reliably threaten the tower without secondary structural support (like the Battle Ram). Therefore, she is classified here as a tank killer rather than a true win condition.   

Battle Ram
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Medium (Charges to Very Fast)
Attack Style	Melee
Targets	Buildings
Combat	HP Level	MEDIUM
DPS Level	Requires Human Decision
Attack Speed	Requires Human Decision
Attack Range	MELEE (Short)
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	True
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	True
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	True
Unique Abilities	None
Strategic	Primary Role	Win Condition
Secondary Roles	Bridge Spam / Lightning Rod
Win Condition	True
Support Card	False
Cycle Card	False
Tags	Generated Tags	troop, ground, melee, win_condition, bridge_spam, building_target, spawner
Structurally, the Battle Ram is a troop consisting of a log carried by two barbarians that exclusively targets enemy buildings. Its base movement speed is medium (60), but it features a defining charge mechanic: after traveling 3 tiles uninterrupted, it accelerates to a very fast speed (120). The Ram structure possesses 967 hitpoints at Level 11. Because it strikes only once before shattering (dealing 286 base damage or 573 charge damage), calculating sustained DPS and attack speed is fundamentally inapplicable to the ram structure itself, necessitating a "Requires Human Decision" marker for standard ML continuous-DPS parsing. The impact occurs at a short melee range of 0.5 tiles and does not produce splash damage.   

The Battle Ram possesses a critical spawning mechanic: upon the Ram's destruction or successful impact, the log shatters and two Barbarians spawn in its place to continue engaging nearby troops or structures. Furthermore, the card features an Evolution variant which incorporates advanced physics mechanics, allowing the evolved ram to push back enemy units and bounce back from certain impacts to charge again. Strategically, the Battle Ram is definitively classified as a win condition due to its strict building-targeting logic. It acts as a primary pressure tool in "Bridge Spam" archetypes, frequently paired with the P.E.K.K.A. to quickly punish low-elixir opponents. Secondarily, the dual-entity nature of the card allows it to act as a "Lightning Rod," absorbing heavy spell strikes to protect more valuable support troops.   

Sparky
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Slow
Attack Style	Ranged Splash
Targets	Ground
Combat	HP Level	Requires Human Decision
DPS Level	Requires Human Decision
Attack Speed	VERY_SLOW
Attack Range	MEDIUM
Splash Damage	True
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	True (Negative Vulnerability)
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	False
Unique Abilities	None
Strategic	Primary Role	Defensive Tank Killer / Area Denial
Secondary Roles	Spell Bait
Win Condition	False
Support Card	True
Cycle Card	False
Tags	Generated Tags	troop, ground, ranged, splash, high_damage, defense
Sparky is a deployable mechanical troop restricted to slow movement and ground-based targeting. Her attack style is characterized by discharging a devastating blast of electricity covering a small area of effect (1.8 tile splash radius) at a medium engagement distance of 5.0 tiles. Exact numerical hitpoints and DPS for Level 11 are omitted from the provided source matrices, requiring a "Requires Human Decision" marker, though textually she is confirmed to possess "high hitpoints" and "extremely high damage" capable of erasing entire pushes instantly. She suffers from the slowest attack speed in the game, requiring exactly 4.0 seconds to charge her coils before firing. Crucially, Sparky features a unique negative interaction regarding reset mechanics: any electric stun applied by the enemy (e.g., Zap, Electro Wizard, Lightning) completely resets her 4-second charge animation back to zero.   

Sparky lacks death damage, spawning mechanics, or an Evolution variant. Strategically, despite possessing unparalleled tower-taking potential if she connects, she is unequivocally disqualified as a true win condition. Because she targets troops, takes 4 seconds to fire, and is easily neutralized by 1-elixir distraction cards or reset spells, she cannot be relied upon to bypass defenses. Instead, she is utilized as a supreme defensive tank killer and area denial tool. She is typically deployed to obliterate heavy ground pushes on the player's side of the arena before acting as a high-threat support piece behind a massive tank like the Goblin Giant. Her presence frequently acts as spell bait, forcing the opponent to aggressively utilize their stun utilities.   

Little Prince
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Medium
Attack Style	Ranged
Targets	Air & Ground
Combat	HP Level	LOW
DPS Level	MEDIUM (Ramps to HIGH)
Attack Speed	FAST (Ramps to VERY_FAST)
Attack Range	LONG
Splash Damage	False
Can Attack Air	True
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False (See Unique)
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	False
Unique Abilities	Champion Ability (Royal Rescue)
Strategic	Primary Role	Ranged Support / Ramp-up Defender
Secondary Roles	Spell Bait
Win Condition	False
Support Card	True
Cycle Card	False
Tags	Generated Tags	troop, champion, ranged, single_target, support, spawner
The Little Prince is an elite Champion-rarity troop moving at a medium pace. He utilizes a repeating crossbow to strike both air and ground targets from a long distance of 5.5 tiles. At Level 11, he is highly fragile, possessing 698 hitpoints (dying fully to a Poison spell but narrowly surviving an equivalent Fireball). His combat metrics are entirely unique due to an accelerating velocity mechanic: his base DPS starts at 86 (medium) but ramps up to 260 (high) as long as he stands still and continues firing. Consequently, his attack speed begins at a fast 1.2 seconds, accelerating rapidly down to a very fast 0.4 seconds. He does not possess innate splash damage or knockback.   

His defining characteristic is his active Champion Ability, "Royal Rescue." For an additional 3 Elixir, the Little Prince summons a secondary melee unit—the Guardienne. Upon deployment, the Guardienne charges forward 4 tiles, dealing 256 burst damage and applying massive physical pushback to enemy units. Following the charge, she operates as an independent melee troop with 1,600 hitpoints. Strategically, the Little Prince is deployed as a ramp-up defender, utilized to shred high-health targets over time from the safety of the backline. The Royal Rescue ability serves as a panic-button to knock away immediate threats and generate an instant tank for a counter-push. He acts as a primary target for medium spells (Spell Bait) to clear the way for allied win conditions. He is strictly a support troop, never a win condition.   

Cannon
Classification	Attribute	Value
Structural	Card Type	Building
Movement	N/A (Stationary)
Attack Style	Ranged
Targets	Ground
Combat	HP Level	MEDIUM
DPS Level	HIGH
Attack Speed	FAST
Attack Range	LONG
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	True
Unique Abilities	Lifetime (30 Seconds)
Strategic	Primary Role	Defensive Building
Secondary Roles	Kiting / Distraction
Win Condition	False
Support Card	False
Cycle Card	False
Tags	Generated Tags	building, ground, ranged, single_target, defense
The Cannon is a stationary structural emplacement that fires single-target projectiles exclusively at ground-based units. At the Level 11 benchmark, the Cannon maintains 824 hitpoints. It acts as a highly efficient source of damage, generating 212 DPS and striking rapidly every 1.0 second. Its engagement radius extends to 5.5 tiles, providing a long reach across the arena. It lacks the capacity to track aerial units, deals no splash damage, and does not push back enemies. As a building, it is inherently bound by a 30-second lifetime, degrading by 27.5 hitpoints per second until it collapses naturally.   

The base Cannon does not deploy secondary units or inflict death damage, but it does feature an Evolution variant introduced in late 2024, altering its baseline mechanics (though specific evolved stats require further granular database expansion). Strategically, the Cannon serves as a premier defensive building. Its primary utility lies in structural manipulation—pulling building-targeting win conditions (like the Hog Rider, Giant, or Golem) toward the center of the arena. This strategic placement forces the enemy tank to absorb fire from both friendly Princess Towers simultaneously while the Cannon leverages its high DPS. Due to its immobility and range constraints, it is utterly incapable of targeting enemy Crown Towers and thus can never operate as a win condition. Though cheap (3 Elixir), it acts as a defensive anchor rather than a true cycle card.   

Royal Delivery
Classification	Attribute	Value
Structural	Card Type	Spell
Movement	N/A
Attack Style	Splash (Spell Impact)
Targets	Air & Ground
Combat	HP Level	N/A
DPS Level	N/A
Attack Speed	N/A
Attack Range	N/A (Radius: 3.0)
Splash Damage	True
Can Attack Air	True
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	True
Spawn Units	True
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	False
Unique Abilities	3.0 Second Deployment Delay
Strategic	Primary Role	Defensive Spell
Secondary Roles	Spawner
Win Condition	False
Support Card	False
Cycle Card	False
Tags	Generated Tags	spell, splash, defense, spawner
The Royal Delivery is structurally classified as a Spell, cast directly from the player's hand to cause an immediate area-of-effect impact. Uniquely, its placement is severely restricted: it can only be cast on the player's own side of the arena. The spell drops a box from the sky, impacting all air and ground units within a 3.0-tile radius. Traditional troop combat metrics (HP, DPS) do not apply to the spell itself. At Level 11, the physical impact of the drop deals 437 splash damage. The card does not generate a knockback or stun effect on impact.   

The defining characteristic of the Royal Delivery is its dual nature as both a spell and a spawner. Upon impact, the box shatters to deploy a single Royal Recruit (a melee unit possessing a health shield) to continue defending. This advantage is balanced by an enormous 3.0-second deploy time delay. This mechanical handicap makes the spell exceptionally difficult to aim, demanding precise, predictive placement to successfully intercept fast-moving threats like the Goblin Barrel. Strategically, because it cannot be deployed across the river, it functions exclusively as a potent defensive mechanism. It is utilized to instantly shatter localized pushes and drop a blocking unit to protect the tower. It cannot target enemy structures and thus cannot serve as a win condition.   

Skeleton Army
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Fast
Attack Style	Melee
Targets	Ground
Combat	HP Level	VERY_LOW
DPS Level	VERY_HIGH (Aggregate)
Attack Speed	FAST
Attack Range	MELEE (Short)
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	True
Unique Abilities	Swarm Generation
Strategic	Primary Role	Swarm Defense / Tank Killer
Secondary Roles	Spell Bait
Win Condition	False
Support Card	False
Cycle Card	False
Tags	Generated Tags	troop, ground, melee, swarm, defense, tank_killer, spell_bait
The Skeleton Army deploys a massive horizontal scatter of 15 physical entities. These units sprint at a fast speed (90) and require point-blank physical contact (0.5 tiles, short melee) to strike ground-based targets. At Level 11, an individual skeleton possesses a mere 81 hitpoints, rendering the entire army highly susceptible to fractional splash damage. However, the army's true strength lies in its aggregate damage; while a single skeleton deals 73 DPS, 15 units swinging simultaneously every 1.1 seconds generates a catastrophic, very high DPS pool capable of instantly melting any isolated ground unit. They possess no splash, charge, or knockback properties.   

The base card does not possess death or spawn damage. The Evolution Ability, however, drastically alters the card's resilience. The Evolved Skeleton Army introduces "General Gerry," a designated skeleton equipped with a hitpoint shield. Crucially, as standard skeletons in the evolved army are destroyed, they respawn as untargetable "shadow skeletons." These shadows continue to deal damage immune to physical strikes until General Gerry himself is defeated. Strategically, the Skeleton Army is deployed defensively to surround and immediately assassinate high-threat, single-target win conditions like the Prince, P.E.K.K.A., or Giant. As a secondary role, it acts as highly efficient "Spell Bait," forcing opponents to expend their Zap or Log, thereby clearing the way for other bait win conditions like the Goblin Barrel. It is completely negated by area-of-effect troops and cannot reliably cross the bridge, firmly excluding it from win condition status.   

Berserker
Classification	Attribute	Value
Structural	Card Type	Troop
Movement	Fast
Attack Style	Melee
Targets	Ground
Combat	HP Level	MEDIUM
DPS Level	MEDIUM
Attack Speed	VERY_FAST
Attack Range	MELEE (Short)
Splash Damage	False
Can Attack Air	False
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	False
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	False
Evolution Ability	False
Unique Abilities	None
Strategic	Primary Role	Swarm Defense
Secondary Roles	Cycle Defense
Win Condition	False
Support Card	True
Cycle Card	True
Tags	Generated Tags	troop, ground, melee, cycle, defense, single_target
The Berserker is a standard ground-targeting troop that sprints at a fast speed (90) to engage enemies physically with cleavers. At Level 11, she possesses 896 hitpoints, allowing her to survive small spell interactions and placing her in the medium durability tier. While her overall 170 DPS is classified as medium, her defining mechanical trait is a very fast attack speed. Striking every 0.6 seconds (with her initial strike registering at an incredibly rapid 0.2 seconds), she attacks faster than nearly any other melee troop in the game. She requires a short melee engagement radius of 0.8 tiles and strictly targets a single ground unit per swing.   

She operates without complex active abilities, lacking death damage, spawn effects, or evolutionary mechanics. Strategically, the Berserker fills a highly specific niche: swarm defense. Due to her ultra-fast attack speed, she can individually chop through swarms like the Goblin Gang or Skeleton Army far more efficiently than standard single-target troops. Costing only 2 Elixir, she provides extreme defensive value while allowing the deck to cycle rapidly back to its core win conditions. She lacks the durability or targeting mechanics to pose a singular threat to enemy structures, operating strictly as a defensive cycle and support card.   

Rage
Classification	Attribute	Value
Structural	Card Type	Spell
Movement	N/A
Attack Style	Splash (Spell Impact)
Targets	Air & Ground
Combat	HP Level	N/A
DPS Level	N/A
Attack Speed	N/A
Attack Range	N/A (Radius: 3.0)
Splash Damage	True
Can Attack Air	True
Can Attack Ground	True
Charge Mechanic	False
Knockback	False
Reset Ability	False
Abilities	Death Damage	False
Spawn Damage	True
Spawn Units	False
Heal	False
Freeze	False
Dash	False
Chain Attack	False
Rage Aura	True
Evolution Ability	False
Unique Abilities	None
Strategic	Primary Role	Support Spell
Secondary Roles	Swarm Clearance
Win Condition	False
Support Card	True
Cycle Card	True
Tags	Generated Tags	spell, splash, support, cycle
Structurally, Rage is a spell cast dynamically onto any location within the arena, instantly encompassing a 3.0-tile radius that impacts both air and ground entities. Traditional hitpoint and DPS metrics are inapplicable. At Level 11, the spell inflicts 179 localized area damage upon deployment.   

The spell's primary mechanic is the Rage Aura. Upon detonation, it leaves behind an area-of-effect field that aggressively accelerates the movement speed and attack speed of all allied units within its borders. It does not spawn units or freeze enemies. Strategically, the 2-Elixir Rage operates exclusively as a support spell. It is heavily utilized in "Beatdown" or "LumberLoon" archetypes to artificially amplify the destructive potential of an active, heavy push (e.g., a Golem or Balloon). Simultaneously, the 179 deployment damage provides massive utility by instantly clearing away fragile defensive swarms (like Skeletons or Bats) attempting to block the push. It must be paired with a physical threat to provide value, disqualifying it as a win condition. Its cheap cost allows it to operate as a cycle card when necessary.   


royaleapi.com
Ronin - New Card - July 2026 (Season 85) - Clash Royale News Blog - RoyaleAPI
Opens in a new window

liquipedia.net
Knight - Liquipedia Clash Royale Wiki
Opens in a new window

clashroyale.fandom.com
Cards | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Category:Troop Cards | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Knight | Clash Royale Wiki - Fandom
Opens in a new window

supercell.com
Season 1 Balance Update! × Clash Royale - Supercell
Opens in a new window

reddit.com
To all new players, this is what a win condition is. Your deck needs 1. I'll be explaining what cards are and are not a win condition since I see lots of arguments about it. : r/ClashRoyale - Reddit
Opens in a new window

clashroyale.fandom.com
Valkyrie | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Evolved Valkyrie - Clash Royale Wiki - Fandom
Opens in a new window

reddit.com
Explanation: Why Valkyrie (Medium Melee) outranges Long Melee units - Reddit
Opens in a new window

clashroyale.fandom.com
Hog Trifecta Control-Cycle | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Firecracker | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Princess Towers | Clash Royale Wiki - Fandom
Opens in a new window

reddit.com
Every clash royale card ranked on a tier list based on how good they are right now (based on the meta/my opinion) : r/RoyaleAPI - Reddit
Opens in a new window

clashroyale.fandom.com
Giant - Clash Royale Wiki - Fandom
Opens in a new window

reddit.com
What are the “win condition” cards? Is there a list somewhere? : r/ClashRoyale - Reddit
Opens in a new window

ldshop.gg
Clash Royale Best Win Conditions Tier List 2026 - LDShop
Opens in a new window

clashroyale.fandom.com
Sparky - Clash Royale Wiki - Fandom
Opens in a new window

royaleapi.com
Miner - Best Decks, Top Players, Battle Stats in Clash Royale - RoyaleAPI
Opens in a new window

clashroyale.fandom.com
Card Overviews | Clash Royale Wiki - Fandom
Opens in a new window

reddit.com
Win condition is a nonsense term with no real definition. Yall really took a term based on vibes and tried to treat it like a concrete thing but it's not : r/ClashRoyale - Reddit
Opens in a new window

clashroyale.fandom.com
User blog:AesDragon/What makes a win condition | Clash Royale Wiki
Opens in a new window

reddit.com
Define a win condition in Clash Royale : r/ClashRoyale - Reddit
Opens in a new window

clashroyale.fandom.com
Mini P.E.K.K.A. - Clash Royale Wiki
Opens in a new window

royaleapi.com
Hero Mini P.E.K.K.A - Best Decks, Top Players, Battle Stats in Clash Royale - RoyaleAPI
Opens in a new window

clashroyale.fandom.com
Skeletons | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Category:Ranger Cards - Clash Royale Wiki
Opens in a new window

clashroyale.fandom.com
Category:Epic Cards - Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
P.E.K.K.A. | Clash Royale Wiki
Opens in a new window

royaleapi.com
P.E.K.K.A - Best Decks, Top Players, Battle Stats in Clash Royale - RoyaleAPI
Opens in a new window

clashroyale.fandom.com
User blog:AesDragon/Hidden card stats: Sight range | Clash Royale Wiki
Opens in a new window

reddit.com
What exactly is a win condition? : r/RoyaleAPI - Reddit
Opens in a new window

discuss.royaleapi.com
Win Condition Debate - General - RoyaleAPI Discuss
Opens in a new window

clashroyale.fandom.com
Battle Ram | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Category:5-Elixir Cards | Clash Royale Wiki - Fandom
Opens in a new window

royaleapi.com
Triple Draft - Best Cards in Clash Royale - RoyaleAPI
Opens in a new window

on.royaleapi.com
Classic Deck: Best Clash Royale Decks - RoyaleAPI
Opens in a new window

clashroyale.fandom.com
Lightning | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Little Prince | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Cannon | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
2.6 Hog - Clash Royale Wiki - Fandom
Opens in a new window

liquipedia.net
Royal Delivery - Liquipedia Clash Royale Wiki
Opens in a new window

clashroyale.fandom.com
Royal Delivery | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Skeleton Army | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Evolved Skeleton Army | Clash Royale Wiki - Fandom
Opens in a new window

clashroyale.fandom.com
Berserker | Clash Royale Wiki - Fandom
Opens in a new window

wikihow.com
Berserker in Clash Royale: Stats & Best Strategies - wikiHow
Opens in a new window

clashroyale.fandom.com
Rage | Clash Royale Wiki - Fandom