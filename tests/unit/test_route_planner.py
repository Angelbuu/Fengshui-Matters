from route_planner import plan_route, replan_route


def test_normal_route_to_radiology():

    route = plan_route(
        start="lobby",
        destination="radiology",
    )

    assert route is not None

    waypoint_ids = [
        waypoint.waypoint_id
        for waypoint in route.waypoints
    ]

    assert waypoint_ids == [
        "corridor_a",
        "corridor_b",
        "corridor_c",
        "radiology",
    ]
    
def test_replan_avoids_blocked_corridor():

    route = replan_route(
        current_location="corridor_a",
        destination="radiology",
        blocked_locations={"corridor_b"},
    )

    assert route is not None

    waypoint_ids = [
        waypoint.waypoint_id
        for waypoint in route.waypoints
    ]

    assert "corridor_b" not in waypoint_ids

    assert waypoint_ids == [
        "corridor_d",
        "corridor_c",
        "radiology",
    ]
    
def test_no_route_when_all_paths_blocked():

    route = replan_route(
        current_location="corridor_a",
        destination="radiology",
        blocked_locations={
            "corridor_b",
            "corridor_d",
        },
    )

    assert route is None