from _deepseek_example_utils import (
    encode_prompts,
    make_llm,
    print_outputs,
    prompt_tokenizer,
    sampling_params,
)


prompts = [
    """
Read the following field report and answer the final question with one short phrase.

Field report: The Silver Orchard water project began as a small repair plan for
three villages on the north side of the valley. The project file says that the
old stone channel was useful during spring melt but unreliable in late summer.
Record A-001 says the first survey team walked from the cedar bridge to the
western pump house and found twelve leaks. Record A-002 says the leaks were not
the biggest problem because the intake screen was blocked by reeds and sand.
Record A-003 says the committee assigned Mira Patel to keep the repair ledger.
Record A-004 says the ledger used blue pages for tools, yellow pages for labor,
and white pages for landowner permissions. Record A-005 says the village of
North Fen sent four carpenters and two masons. Record A-006 says the village of
Hillside sent a cart, two oxen, and a cook named Pavel.

Record B-001 says the first temporary bridge was called Willow Bridge. It was
built from pine boards and rope, and it was intended only for foot traffic.
Record B-002 says Willow Bridge washed out after three days of heavy rain.
Record B-003 says a second temporary bridge was built with iron pins, ash
beams, and a gravel approach. The second bridge was named Hawthorn Bridge in
the ledger because hawthorn trees marked the crossing. Record B-004 says the
committee wrote the name Hawthorn Bridge on every delivery receipt after the
flood. Record B-005 says the bridge carried lime, pump parts, sacks of oats,
and the spare intake screen. Record B-006 says the final stone bridge would not
be started until the following year.

Record C-001 says the engineer Elena Ruiz changed the pump schedule after the
third inspection. Record C-002 says the morning pump test began at 06:20 and
ended at 09:10. Record C-003 says the evening pump test began at 17:30 and
ended at 19:00. Record C-004 says the evening test produced cleaner water but
less pressure. Record C-005 says the repair team kept the old wooden valve
because the replacement valve was too wide. Record C-006 says the spare valve
was stored in the tool shed and tagged with red twine. Record C-007 says the
red twine tag should not be confused with the yellow paint marks on survey
stakes.

Record D-001 says village elders wanted the channel finished before the barley
harvest. Record D-002 says the harvest date was uncertain because the valley
had received uneven rain. Record D-003 says the south fields were too wet for
wagons, while the north fields were dry enough for threshing. Record D-004 says
the orchard owners promised apples for the work crew if water reached the upper
cistern by Monday. Record D-005 says the upper cistern did receive water by
Monday afternoon, but the pressure remained low. Record D-006 says the crew
therefore recorded the milestone as partial, not complete.

Record E-001 says the final inspection was held in the schoolhouse. Record
E-002 says the committee asked three questions: whether the channel leaked,
whether the pump could run for two hours, and whether the temporary bridge was
safe for supply carts. Record E-003 says the channel still leaked near marker
seventeen. Record E-004 says the pump ran for two hours after the intake screen
was cleaned. Record E-005 says the temporary bridge was safe for carts only if
the carts crossed one at a time. Record E-006 says the bridge name in the final
inspection notes was Hawthorn Bridge, not Willow Bridge. Record E-007 says
Mira Patel signed the notes, and Elena Ruiz countersigned them.

Record F-001 says a later clerk added a confusing margin note that mentioned
Willow Bridge beside the final inspection. Record F-002 says the margin note
was copied from the first week of repairs and should not override the signed
inspection notes. Record F-003 says the archive index lists Hawthorn Bridge as
the active temporary supply crossing during the final inspection. Record F-004
says the final public notice thanked the carpenters for rebuilding the crossing
after the flood. Record F-005 says no document after the flood used Willow
Bridge as the official crossing name. Record F-006 says Hawthorn Bridge stayed
in use until the permanent stone bridge opened.

Appendix note G-001 says the orchard committee also tracked rope, nails,
chalk, canvas, lamp oil, spare boots, survey flags, and meal tickets. Appendix
note G-002 says none of those supplies changed the bridge name. Appendix note
G-003 says the blue ledger pages listed hammers, augers, chisels, saw teeth,
mallets, clamps, buckets, and wedges. Appendix note G-004 says the yellow
ledger pages listed digging crews, carpenters, cooks, watchmen, drivers, and
night guards. Appendix note G-005 says the white ledger pages listed field
edges, orchard gates, schoolyard access, and the lane beside the mill. Appendix
note G-006 says a broken wheel delayed the lime cart, but the cart still
crossed Hawthorn Bridge. Appendix note G-007 says the oxen were rested beside
the hawthorn trees. Appendix note G-008 says the spare intake screen was
wrapped in canvas before it crossed the temporary bridge. Appendix note G-009
says the cook Pavel counted meals near the crossing. Appendix note G-010 says
Mira Patel marked the bridge receipts with a small triangle.

Appendix note G-011 says the pump house roof leaked during the same week.
Appendix note G-012 says the roof leak was repaired after the water channel.
Appendix note G-013 says the elders did not include the roof leak in the final
bridge decision. Appendix note G-014 says the old cedar bridge remained closed
to carts. Appendix note G-015 says Willow Bridge remained listed only in early
drafts. Appendix note G-016 says Hawthorn Bridge appeared on the final supply
map. Appendix note G-017 says the final supply map was folded into the back of
the signed inspection packet. Appendix note G-018 says Elena Ruiz checked the
map against the cart receipts. Appendix note G-019 says the cart receipts
matched Hawthorn Bridge. Appendix note G-020 says the schoolhouse copy also
matched Hawthorn Bridge.

Appendix note G-021 says the committee used red ink for hazards and black ink
for names. Appendix note G-022 says Hawthorn Bridge was written in black ink.
Appendix note G-023 says a red circle marked the washed-out Willow Bridge.
Appendix note G-024 says the red circle did not mean Willow Bridge was active.
Appendix note G-025 says the public notice avoided the old name to prevent
confusion. Appendix note G-026 says the route from North Fen to the upper
cistern crossed Hawthorn Bridge twice. Appendix note G-027 says the route from
Hillside crossed the same bridge once. Appendix note G-028 says the inspector
walked the bridge before allowing loaded carts. Appendix note G-029 says the
loaded carts crossed one at a time. Appendix note G-030 says the inspector's
summary called the crossing Hawthorn Bridge.

Appendix note G-031 says the archive later received a clean typed copy.
Appendix note G-032 says the typed copy preserved the signed inspection
wording. Appendix note G-033 says the typed copy did not preserve the mistaken
margin note. Appendix note G-034 says a volunteer indexer added both bridge
names as search terms. Appendix note G-035 says the search term Willow Bridge
points to the flood damage file. Appendix note G-036 says the search term
Hawthorn Bridge points to the final inspection file. Appendix note G-037 says
the permanent stone bridge was outside the scope of the temporary inspection.
Appendix note G-038 says the temporary bridge in the question is the bridge
used before the stone bridge. Appendix note G-039 says that temporary bridge
was Hawthorn Bridge. Appendix note G-040 says the answer should use the bridge
name from the signed inspection notes.

Audit trail G-041: HAWTHORN-A1-0001 HAWTHORN-A1-0002 HAWTHORN-A1-0003
HAWTHORN-A1-0004 HAWTHORN-A1-0005 HAWTHORN-A1-0006. Audit trail G-042:
BRIDGE-FINAL-ALPHA BRIDGE-FINAL-BETA BRIDGE-FINAL-GAMMA BRIDGE-FINAL-DELTA.
Audit trail G-043: RECEIPT-CART-17 RECEIPT-CART-18 RECEIPT-CART-19
RECEIPT-CART-20 RECEIPT-CART-21. Audit trail G-044: INSPECTION-SIGNED-MIRA
INSPECTION-SIGNED-ELENA INSPECTION-SCHOOLHOUSE-COPY. Audit trail G-045:
WILLOW-DAMAGE-FILE WILLOW-FLOOD-NOTE WILLOW-OLD-DRAFT WILLOW-NOT-ACTIVE.
Audit trail G-046: HAWTHORN-ACTIVE-CROSSING HAWTHORN-SUPPLY-MAP
HAWTHORN-CART-ROUTE HAWTHORN-FINAL-NOTE. Audit trail G-047:
PUMP-TEST-MORNING PUMP-TEST-EVENING PUMP-TEST-CLEAN-SCREEN.
Audit trail G-048: LEDGER-BLUE-TOOLS LEDGER-YELLOW-LABOR
LEDGER-WHITE-PERMISSION. Audit trail G-049: CISTERN-UPPER-MONDAY
CISTERN-PARTIAL-MILESTONE CISTERN-LOW-PRESSURE. Audit trail G-050:
ANSWER-BRIDGE-HAWTHORN ANSWER-BRIDGE-HAWTHORN ANSWER-BRIDGE-HAWTHORN.
Calibration row G-051: HAWTHORN-FINAL-001 HAWTHORN-FINAL-002
HAWTHORN-FINAL-003 HAWTHORN-FINAL-004 HAWTHORN-FINAL-005 HAWTHORN-FINAL-006
HAWTHORN-FINAL-007 HAWTHORN-FINAL-008 HAWTHORN-FINAL-009 HAWTHORN-FINAL-010.
Calibration row G-052: SUPPLY-CROSSING-011 SUPPLY-CROSSING-012
SUPPLY-CROSSING-013 SUPPLY-CROSSING-014 SUPPLY-CROSSING-015
SUPPLY-CROSSING-016 SUPPLY-CROSSING-017 SUPPLY-CROSSING-018.
Calibration row G-053: INSPECTED-BRIDGE-019 INSPECTED-BRIDGE-020
INSPECTED-BRIDGE-021 INSPECTED-BRIDGE-022 INSPECTED-BRIDGE-023
INSPECTED-BRIDGE-024 INSPECTED-BRIDGE-025 INSPECTED-BRIDGE-026.

Question: What was the name of the temporary bridge used during the final
inspection?
""",
    """
Read the following museum archive notes and answer the final question with one
short sentence.

Archive note 01: The Lanton Museum prepared an exhibition about winter trade
routes, river markets, and portable lamps. The preparation binder contains
shipping forms, conservation notes, school-program drafts, and a catalog of
objects. Archive note 02: The exhibition team met every Tuesday in Gallery
Four, but the object catalog was usually reviewed in the north reading room.
Archive note 03: The first draft grouped objects by material: brass, tin,
painted wood, leather, glass, paper, and wool. Archive note 04: The second
draft grouped objects by use: navigation, storage, cooking, clothing, lighting,
and trade records. Archive note 05: The final catalog used the second grouping.

Archive note 06: The lighting section included six lanterns, two candle boxes,
three oil tins, and a travel mirror. Archive note 07: Lantern L-14 had a blue
glass panel and a dented brass hook. Archive note 08: Lantern L-22 had a
replacement handle made by a blacksmith in Orley. Archive note 09: Lantern
L-31 had a paper tag that said "river fog, east bank." Archive note 10:
Lantern L-38 had a cracked hinge and was removed from public display. Archive
note 11: Lantern L-43 was accepted because its wick box was original. Archive
note 12: Lantern L-51 was rejected because it was a modern theater prop.

Archive note 13: The curatorial team was led by Rowan Ellis. Archive note 14:
The registrar was June Calder, and the conservator was Tomas Venn. Archive
note 15: The education officer was Priya Sen. Archive note 16: The director
was Maren Holt, but she did not approve individual object records. Archive
note 17: Rowan Ellis approved the lantern catalog on 14 February. Archive note
18: June Calder approved the shipping crate list on 15 February. Archive note
19: Tomas Venn approved the cleaning instructions on 16 February. Archive note
20: Priya Sen approved the school handout on 17 February.

Archive note 21: A volunteer later wrote that June Calder had approved the
lantern catalog. Archive note 22: The staff correction says this was wrong:
June Calder approved shipping paperwork, not the lantern catalog. Archive note
23: The correction repeats that Rowan Ellis approved the lantern catalog.
Archive note 24: The correction also says that Rowan Ellis changed the title of
the lighting section from "Lamps and Flame" to "Portable Light." Archive note
25: The museum label for Lantern L-22 credits the Orley blacksmith but does not
name the approving curator. Archive note 26: The internal catalog does name
Rowan Ellis on the approval line.

Archive note 27: The exhibition opened with three public talks. The first talk
described river fog and night travel. The second talk described winter
markets. The third talk described repair marks on common objects. Archive note
28: A newspaper review praised the blue glass of Lantern L-14. Archive note
29: A school group asked why Lantern L-38 was not displayed. Archive note 30:
Tomas Venn explained that the cracked hinge was too fragile. Archive note 31:
The museum shop sold a postcard of Lantern L-31. Archive note 32: The
postcard caption was written by Priya Sen.

Archive note 33: The closing checklist says all lanterns returned to storage
except L-14, which stayed out for photography. Archive note 34: Rowan Ellis
signed the closing checklist as curator. Archive note 35: June Calder signed
the loan return sheet as registrar. Archive note 36: Tomas Venn signed the
condition report as conservator. Archive note 37: Priya Sen signed the school
program summary as education officer. Archive note 38: Maren Holt signed the
annual report as director. Archive note 39: None of those later signatures
changes the approval line in the object catalog.

Archive note 40: The archive index has four cross references. Cross reference
A points from "lighting objects" to "Portable Light." Cross reference B points
from "Orley handle" to Lantern L-22. Cross reference C points from "catalog
approval" to Rowan Ellis. Cross reference D points from "shipping approval" to
June Calder. Archive note 41: The index was made after the exhibition closed
and is considered reliable because it was checked against signed forms.
Archive note 42: The final summary says the lantern catalog was approved by
Rowan Ellis, the shipping list by June Calder, the conservation notes by Tomas
Venn, and the school handout by Priya Sen.

Archive note 43: The storage appendix lists shelf marks, object trays,
humidity cards, label drafts, and loan copies. Archive note 44: Shelf mark
P-01 contains Lantern L-14 and a spare blue glass note. Archive note 45: Shelf
mark P-02 contains Lantern L-22 and the Orley handle receipt. Archive note 46:
Shelf mark P-03 contains Lantern L-31 and the river fog tag. Archive note 47:
Shelf mark P-04 contains the removed Lantern L-38 and a cracked hinge warning.
Archive note 48: Shelf mark P-05 contains Lantern L-43 and the original wick
box note. Archive note 49: Shelf mark P-06 contains the rejected prop L-51.
Archive note 50: None of the shelf marks changes the catalog approval line.

Archive note 51: The object photography log says Rowan Ellis selected the
angle for Lantern L-14. Archive note 52: The same log says June Calder checked
the crate number after photography. Archive note 53: The same log says Tomas
Venn inspected the hinge before the lamp was moved. Archive note 54: The same
log says Priya Sen requested one photograph for the school handout. Archive
note 55: The photography log was created after the lantern catalog was already
approved. Archive note 56: The approval name remained Rowan Ellis. Archive
note 57: The approval date remained 14 February. Archive note 58: The section
title remained Portable Light. Archive note 59: The old title Lamps and Flame
was not restored. Archive note 60: The lighting section remained grouped by
use, not by material.

Archive note 61: The audio guide script contains a summary of each staff role.
Archive note 62: It says Rowan Ellis was curator for the exhibition. Archive
note 63: It says June Calder was registrar for loans and shipping. Archive
note 64: It says Tomas Venn was conservator for cleaning and condition.
Archive note 65: It says Priya Sen was education officer for school materials.
Archive note 66: It says Maren Holt was director for annual reporting. Archive
note 67: The audio guide does not contradict the catalog. Archive note 68:
The audio guide says the lantern catalog was approved by the curator. Archive
note 69: The named curator in the archive notes is Rowan Ellis. Archive note
70: Therefore the approval line points to Rowan Ellis.

Archive note 71: The donor file contains letters about winter markets, river
fog, damaged handles, tin oil cans, and display lighting. Archive note 72: The
donor file thanks June Calder for arranging transport. Archive note 73: The
donor file thanks Tomas Venn for stabilizing fragile objects. Archive note 74:
The donor file thanks Priya Sen for public programs. Archive note 75: The
donor file thanks Rowan Ellis for catalog approval and object selection.
Archive note 76: The donor file was checked against the internal catalog.
Archive note 77: The internal catalog is the controlling document for the
question. Archive note 78: The controlling document names Rowan Ellis. Archive
note 79: No signed form names June Calder as lantern catalog approver. Archive
note 80: The answer should name Rowan Ellis.

Archive note 81: Catalog checksum row C-001 lists ROWAN-ELLIS-CURATOR,
ROWAN-ELLIS-APPROVAL, ROWAN-ELLIS-LANTERN-CATALOG. Archive note 82:
Catalog checksum row C-002 lists JUNE-CALDER-SHIPPING, JUNE-CALDER-CRATE,
JUNE-CALDER-REGISTRAR. Archive note 83: Catalog checksum row C-003 lists
TOMAS-VENN-CONSERVATION, TOMAS-VENN-HINGE, TOMAS-VENN-CLEANING. Archive note
84: Catalog checksum row C-004 lists PRIYA-SEN-SCHOOL, PRIYA-SEN-HANDOUT,
PRIYA-SEN-POSTCARD. Archive note 85: Catalog checksum row C-005 lists
MAREN-HOLT-DIRECTOR, MAREN-HOLT-ANNUAL, MAREN-HOLT-NOT-CATALOG. Archive note
86: Lantern index row L-001 says LANTERN-L14-BLUE-GLASS and CURATOR-ROWAN.
Archive note 87: Lantern index row L-002 says LANTERN-L22-ORLEY-HANDLE and
CURATOR-ROWAN. Archive note 88: Lantern index row L-003 says LANTERN-L31-RIVER
and CURATOR-ROWAN. Archive note 89: Lantern index row L-004 says LANTERN-L38
REMOVED and CONSERVATOR-TOMAS. Archive note 90: Answer checksum row C-010
says APPROVER-ROWAN-ELLIS, APPROVER-ROWAN-ELLIS, APPROVER-ROWAN-ELLIS.
Archive note 91: Catalog row R-001 says ROWAN-CATALOG-001 ROWAN-CATALOG-002
ROWAN-CATALOG-003 ROWAN-CATALOG-004 ROWAN-CATALOG-005 ROWAN-CATALOG-006
ROWAN-CATALOG-007 ROWAN-CATALOG-008 ROWAN-CATALOG-009 ROWAN-CATALOG-010.
Archive note 92: Catalog row R-002 says LANTERN-APPROVAL-011
LANTERN-APPROVAL-012 LANTERN-APPROVAL-013 LANTERN-APPROVAL-014
LANTERN-APPROVAL-015 LANTERN-APPROVAL-016 LANTERN-APPROVAL-017.
Archive note 93: Catalog row R-003 says CURATOR-ROWAN-018 CURATOR-ROWAN-019
CURATOR-ROWAN-020 CURATOR-ROWAN-021 CURATOR-ROWAN-022 CURATOR-ROWAN-023.

Question: Who approved the lantern catalog?
""",
    """
Read the following incident chronology and answer the final question with one
short phrase.

Chronology entry 001: The coastal radio station at Bell Point ran a drill for
storm communications. The drill simulated a broken landline, a flooded road,
and a delayed fuel truck. Chronology entry 002: The staff used three message
channels: a shortwave radio, a harbor signal lamp, and a courier boat.
Chronology entry 003: The shortwave radio was tested first because it could
reach the inland emergency office. Chronology entry 004: The signal lamp was
tested second because it could reach the harbor master. Chronology entry 005:
The courier boat was tested last because it required daylight.

Chronology entry 006: The station chief was Dana Moore. The radio operator was
Felix Arno. The harbor contact was Nia Bell. The courier boat captain was
Jonas Reed. Chronology entry 007: Dana Moore wrote the master schedule. Felix
Arno wrote the radio log. Nia Bell wrote the harbor receipt. Jonas Reed wrote
the boat delivery note. Chronology entry 008: The master schedule says the
first code word was "amber." Chronology entry 009: The radio log says the
second code word was "cedar." Chronology entry 010: The harbor receipt says
the third code word was "lantern." Chronology entry 011: The delivery note
says the fourth code word was "anchor."

Chronology entry 012: During the morning drill, the shortwave antenna slipped
out of alignment. Felix Arno corrected the angle and repeated the message.
Chronology entry 013: The repeated message used the code word "cedar" and was
received clearly. Chronology entry 014: The harbor signal lamp had no
mechanical fault, but fog reduced visibility. Chronology entry 015: Nia Bell
confirmed the lamp message after the fog thinned. Chronology entry 016: The
courier boat left late because the tide was low. Chronology entry 017: Jonas
Reed delivered the packet after noon and marked it as delayed but complete.

Chronology entry 018: The afternoon review compared the three channels.
Chronology entry 019: The shortwave radio was rated fastest after the antenna
was corrected. Chronology entry 020: The harbor signal lamp was rated useful
only in fair visibility. Chronology entry 021: The courier boat was rated slow
but reliable for physical documents. Chronology entry 022: Dana Moore wrote
that the backup channel for urgent messages should be the shortwave radio.
Chronology entry 023: Dana Moore wrote that the backup channel for signed
paperwork should be the courier boat. Chronology entry 024: Dana Moore wrote
that the signal lamp should remain a local harbor channel, not the primary
backup for urgent inland messages.

Chronology entry 025: A later training card introduced confusion by saying
"use the lamp when the landline fails." Chronology entry 026: The correction
sheet says the card was shortened for trainees and omitted the distinction
between harbor messages and inland emergency messages. Chronology entry 027:
The correction sheet says urgent inland messages should use the shortwave
radio when the landline fails. Chronology entry 028: The correction sheet says
signed paper documents should use the courier boat when the road is flooded.
Chronology entry 029: The correction sheet says the signal lamp is only a
harbor confirmation channel.

Chronology entry 030: The final drill report lists four lessons. Lesson one:
align the shortwave antenna before sending the second code word. Lesson two:
keep spare lamp oil in the harbor cabinet. Lesson three: load the courier boat
before the tide falls. Lesson four: teach staff that different backups serve
different message types. Chronology entry 031: The final drill report states
again that the shortwave radio is the backup for urgent inland emergency
messages. Chronology entry 032: The final drill report states again that the
courier boat is the backup for signed paperwork. Chronology entry 033: The
final drill report states again that the signal lamp is useful for harbor
confirmation but not enough for inland emergency traffic.

Chronology entry 034: In the closing interview, Dana Moore said the most common
mistake was treating every backup channel as interchangeable. Chronology entry
035: Felix Arno said the shortwave set worked after he tightened the antenna
bracket. Chronology entry 036: Nia Bell said the lamp was easy to read once
the fog lifted. Chronology entry 037: Jonas Reed said the boat was dependable
but not quick. Chronology entry 038: The station filed the drill under "storm
communications, Bell Point, urgent inland backup." Chronology entry 039: The
index term "urgent inland backup" points to shortwave radio. Chronology entry
040: The index term "signed paperwork backup" points to courier boat.

Chronology entry 041: The equipment appendix lists batteries, dry cloths,
signal flags, lamp oil, antenna bolts, spare fuses, waterproof envelopes, and
fuel coupons. Chronology entry 042: The batteries supported the shortwave set.
Chronology entry 043: The lamp oil supported the harbor signal lamp.
Chronology entry 044: The waterproof envelopes supported the courier boat.
Chronology entry 045: The appendix warns that equipment categories should not
be confused with message priorities. Chronology entry 046: Urgent inland
messages have the highest priority. Chronology entry 047: Signed paperwork has
a lower speed priority but requires physical delivery. Chronology entry 048:
Harbor confirmation messages are local and visual. Chronology entry 049: The
appendix repeats that urgent inland messages use the shortwave radio when the
landline fails. Chronology entry 050: The appendix repeats that paperwork uses
the courier boat when roads fail.

Chronology entry 051: The second training session tested the same distinction.
Chronology entry 052: Trainee A chose the lamp for an inland medical alert and
was corrected. Chronology entry 053: Trainee B chose the shortwave radio for
the same alert and passed. Chronology entry 054: Trainee C chose the courier
boat for a signed fuel form and passed. Chronology entry 055: Trainee D chose
the signal lamp for harbor arrival confirmation and passed. Chronology entry
056: The instructor wrote that channel choice depends on message type.
Chronology entry 057: The instructor wrote that speed matters most for urgent
inland emergency messages. Chronology entry 058: The instructor wrote that the
shortwave radio is the fastest usable backup after antenna alignment.
Chronology entry 059: The instructor wrote that the boat is reliable but slow.
Chronology entry 060: The instructor wrote that the lamp is visible only in
fair harbor conditions.

Chronology entry 061: The station checklist contains a line labeled LANDLINE
FAILED. Chronology entry 062: Under that line, urgent inland alert points to
shortwave radio. Chronology entry 063: Under that line, signed documents point
to courier boat. Chronology entry 064: Under that line, harbor acknowledgement
points to signal lamp. Chronology entry 065: The checklist was signed by Dana
Moore. Chronology entry 066: The checklist was reviewed by Felix Arno.
Chronology entry 067: The checklist was copied into the final drill packet.
Chronology entry 068: The final drill packet is the source named in the
question. Chronology entry 069: The final drill packet confirms shortwave
radio for urgent inland emergency messages. Chronology entry 070: The final
drill packet does not assign that role to the signal lamp.

Chronology entry 071: A harbor newsletter later simplified the story and said
"the lamp saved the drill." Chronology entry 072: The station archivist marked
that newsletter as colorful but incomplete. Chronology entry 073: The
archivist wrote that the lamp helped harbor confirmation only. Chronology
entry 074: The archivist wrote that the shortwave radio handled inland
emergency backup. Chronology entry 075: The archivist wrote that the courier
boat handled physical paperwork backup. Chronology entry 076: The archive
index follows the final drill report, not the newsletter. Chronology entry
077: The final drill report follows Dana Moore's schedule and correction
sheet. Chronology entry 078: Dana Moore's schedule separates urgent messages
from paperwork. Chronology entry 079: The relevant urgent inland backup is the
shortwave radio. Chronology entry 080: The answer should be shortwave radio.

Chronology entry 081: Drill checksum row D-001 lists SHORTWAVE-URGENT-INLAND,
SHORTWAVE-LANDLINE-FAIL, SHORTWAVE-FAST-BACKUP. Chronology entry 082: Drill
checksum row D-002 lists COURIER-SIGNED-PAPER, COURIER-ROAD-FLOOD,
COURIER-SLOW-RELIABLE. Chronology entry 083: Drill checksum row D-003 lists
SIGNAL-LAMP-HARBOR, SIGNAL-LAMP-FOG-LIMIT, SIGNAL-LAMP-NOT-INLAND. Chronology
entry 084: Drill checksum row D-004 lists DANA-MOORE-SCHEDULE,
FELIX-ARNO-RADIO, NIA-BELL-HARBOR, JONAS-REED-BOAT. Chronology entry 085:
Checklist row L-001 says LANDLINE-FAILED-URGENT-INLAND-SHORTWAVE. Chronology
entry 086: Checklist row L-002 says LANDLINE-FAILED-SIGNED-FORM-COURIER.
Chronology entry 087: Checklist row L-003 says LANDLINE-FAILED-HARBOR-LAMP.
Chronology entry 088: Archive row A-001 says FINAL-REPORT-SHORTWAVE. Chronology
entry 089: Archive row A-002 says NEWSLETTER-LAMP-INCOMPLETE. Chronology entry
090: Answer checksum row A-010 says SHORTWAVE-RADIO, SHORTWAVE-RADIO,
SHORTWAVE-RADIO.
Chronology entry 091: Radio checksum R-001 says SHORTWAVE-URGENT-001
SHORTWAVE-URGENT-002 SHORTWAVE-URGENT-003 SHORTWAVE-URGENT-004
SHORTWAVE-URGENT-005 SHORTWAVE-URGENT-006 SHORTWAVE-URGENT-007.
Chronology entry 092: Radio checksum R-002 says LANDLINE-FAILED-008
LANDLINE-FAILED-009 LANDLINE-FAILED-010 LANDLINE-FAILED-011
LANDLINE-FAILED-012 LANDLINE-FAILED-013 LANDLINE-FAILED-014.
Chronology entry 093: Radio checksum R-003 says INLAND-BACKUP-015
INLAND-BACKUP-016 INLAND-BACKUP-017 INLAND-BACKUP-018 INLAND-BACKUP-019
INLAND-BACKUP-020.

Question: According to the final drill report, what backup channel should be
used for urgent inland emergency messages when the landline fails?
""",
]


if __name__ == "__main__":
    llm = make_llm(
        max_model_len=4096,
        max_num_prefill_seqs_per_step=1,
        max_num_decode_seqs_per_step=3,
    )
    tokenizer = prompt_tokenizer(llm)
    prompt_token_ids = encode_prompts(tokenizer, prompts)
    for i, ids in enumerate(prompt_token_ids, 1):
        print(f"long prompt {i} token_len={len(ids)}")
    outputs = llm.generate(
        prompt_token_ids,
        sampling_params(max_tokens=32),
    )
    print_outputs(prompts, prompt_token_ids, outputs)
