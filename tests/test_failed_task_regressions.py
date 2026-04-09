from agent import Agent, RESPOND_ACTION_NAME


def test_task14_balance_request_uses_profile_balances():
    agent = Agent()
    agent._update_state_from_tool_payload(
        'tool: {"user_id":"mohamed_silva_9265","payment_methods":{"gift_card_8020792":{"id":"gift_card_8020792","amount":198},"certificate_3765853":{"id":"certificate_3765853","amount":500},"credit_card_2198526":{"id":"credit_card_2198526"}}}'
    )
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "What are the balances on my gift cards and certificates?"',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "gift_card_8020792" in action["arguments"]["content"]
    assert "certificate_3765853" in action["arguments"]["content"]
    assert "credit_card_2198526" not in action["arguments"]["content"]


def test_task20_selected_flight_number_is_not_treated_as_reservation_id():
    agent = Agent()
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "I would like to book flight HAT083."',
        }
    )

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {"reservation_id": "HAT083"}}
    )

    assert action["name"] == RESPOND_ACTION_NAME
    assert "flight number" in action["arguments"]["content"]


def test_task22_can_reuse_reservation_inventory_for_other_flight_lookup():
    agent = Agent()
    agent.session_state["reservation_id"] = "D1EW9B"
    agent.session_state["reservation_inventory"] = {
        "D1EW9B": {
            "origin": "ATL",
            "destination": "JFK",
            "dates": ["2024-05-17"],
            "created_at": "2024-05-04T07:38:29",
            "status": None,
            "passenger_count": 1,
        },
        "9HBUV8": {
            "origin": "DEN",
            "destination": "BOS",
            "dates": ["2024-05-17"],
            "created_at": "2024-05-12T17:08:42",
            "status": None,
            "passenger_count": 1,
        },
    }
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "Please check my other flight on 2024-05-17."',
        }
    )

    action = agent._guard_action(
        {"name": "get_reservation_details", "arguments": {}}
    )

    assert action == {
        "name": "get_reservation_details",
        "arguments": {"reservation_id": "9HBUV8"},
    }


def test_task39_opening_turn_cancel_all_upcoming_starts_with_profile_lookup():
    agent = Agent()
    agent.turn_count = 1
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "My user id is amelia_davis_8890. Yes, I want to cancel all my upcoming flights."',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "amelia_davis_8890"},
    }


def test_task43_successful_mutation_adds_post_tool_summary_hint():
    agent = Agent()
    agent.session_state["last_tool_name"] = "cancel_reservation"
    agent.messages.append(
        {
            "role": "user",
            "content": 'tool: {"reservation_id":"9HBUV8","status":"cancelled"}',
        }
    )

    messages = agent._messages_for_model()
    joined = "\n".join(
        msg["content"] for msg in messages if isinstance(msg.get("content"), str)
    )

    assert "previous tool call succeeded" in joined.lower()


def test_task44_balance_intent_with_user_id_prefers_lookup_over_refusal():
    agent = Agent()
    agent.turn_count = 1
    agent.messages.append(
        {
            "role": "user",
            "content": 'User message: "My user ID is sophia_silva_7557. Can you tell me my certificate balances?"',
        }
    )

    action = agent._guard_action(
        {"name": "respond", "arguments": {"content": "placeholder"}}
    )

    assert action == {
        "name": "get_user_details",
        "arguments": {"user_id": "sophia_silva_7557"},
    }
