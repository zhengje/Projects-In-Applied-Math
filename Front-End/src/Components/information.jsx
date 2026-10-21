export default function Information({ rank, title, price, rating}) {
    return (
    <div>
        { price > 0 ? (
        <li>
                RANK {rank}. {title} | ${price}
        </li>
        ) : (
        <li>
                RANK {rank}. {title} | Free
        </li>
        )
        }

    </div>
    )
}